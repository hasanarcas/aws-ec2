import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as IMAGE
from flask import Flask, request, jsonify
import pickle
from googletrans import Translator

translator = Translator()
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))
        
        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)
        
        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
    
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
    
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)
        
    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def load_models():
    with open('./model/tokenizer_17.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    encoder = CNN_Encoder(256)
    sample_input = tf.ones((64, 49, 2048))
    encoder(sample_input)
    encoder_weights = np.load('./model/encoder_weights_17.npy', allow_pickle=True)
    encoder.set_weights(encoder_weights)

    decoder = RNN_Decoder(256, 512, 5001)
    decoder(tf.ones((64, 1)), tf.ones((64, 49, 256)), decoder.reset_state(batch_size=64))
    decoder_weights = np.load('./model/decoder_weights_17.npy', allow_pickle=True)
    decoder.set_weights(decoder_weights)

    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    sample_input = tf.ones((1, 299, 299, 3))
    image_features_extract_model(sample_input)
    image_features_extract_model_weights = np.load('./model/image_features_extract_model_weights_17.npy', allow_pickle=True)
    image_features_extract_model.set_weights(image_features_extract_model_weights)
    return tokenizer, encoder, decoder, image_features_extract_model

tokenizer, encoder, decoder, image_features_extract_model = load_models()

def load_image(image_path):
   img = tf.io.read_file(image_path)
   img = tf.image.decode_jpeg(img, channels=3)
   img = tf.image.resize(img, (299, 299))
   img = tf.keras.applications.inception_v3.preprocess_input(img)
   return img, image_path

def evaluate(image):
   hidden = decoder.reset_state(batch_size=1) #save_this decoder
   temp_input = tf.expand_dims(load_image(image)[0], 0)
   img_tensor_val = image_features_extract_model(temp_input)
   img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

   features = encoder(img_tensor_val) #save_this encoder
   dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
   result = []

   for i in range(47): 
       predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
       predicted_id = tf.argmax(predictions[0]).numpy()  # tf.random.categorical(predictions, 1)[0][0].numpy()
       result.append(tokenizer.index_word[predicted_id])

       if tokenizer.index_word[predicted_id] == '<end>':
           return result

       dec_input = tf.expand_dims([predicted_id], 0)
        
   return result

def caption_this_image(image):
   result = evaluate(image)
   for i in result:
    if i=="<unk>":
        result.remove(i)
    result_join = ' '.join(result)
    result_final = result_join.rsplit(' ', 1)[0]
    result_final = translator.translate(result_final, dest='tr').text
    print("Final Result is ------------------>>>>>>>>>>>>>>>" + result_final)
    return result_final

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return jsonify(error='No file found')
        
    file = request.files.get('file')
    img_bytes = file.read()
    img_path = './aws/upload_image/test.jpg'
    with(open(img_path, 'wb')) as img:
        img.write(img_bytes)
    result = caption_this_image(img_path)
    return jsonify(prediction=result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')