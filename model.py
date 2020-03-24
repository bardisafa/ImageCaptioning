#from __future__ import print_function
# importing modules
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import array
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout,\
                         BatchNormalization, Concatenate, GRU
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img,img_to_array
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import csv
from pickle import dump, load


# Creating a dictionary to hold 300dim value for all words in training caption
def read_dictionary():
    csvfile = open('encoded dictionary.csv', newline='')
    csvreader = csv.reader(csvfile)

    encoded_dictionary = []
    for row in csvreader:
        encoded_dictionary = encoded_dictionary + [row]
    encoded_dictionary = encoded_dictionary[1:]

    embedding_dim = 300
    vocab_size = len(encoded_dictionary)
    embedding_matrix = np.zeros((vocab_size , embedding_dim))

    embeddings_dictionary={}
    for i in range(len(encoded_dictionary)):
        embeddings_dictionary[encoded_dictionary[i][0]] = np.array(encoded_dictionary[i][1:])

    for i in range(len(encoded_dictionary)):
        embedding_vector = np.array(encoded_dictionary[i][1:])
        embedding_matrix[i] = embedding_vector
    return embedding_matrix, embeddings_dictionary

embedding_matrix, embeddings_dictionary = read_dictionary()


def find_word2ix(embeddings_dictionary):
    word2ix={}
    #this dictionary contains index as key and words as value
    ix2word={}
    ix=0
    for w in list(embeddings_dictionary.keys()):
        word2ix[w]=ix
        ix2word[ix]=w
        ix+=1
    return word2ix, ix2word
word2ix, ix2word = find_word2ix(embeddings_dictionary)


def read_edited_caption():
    csvfile = open('edited caption.csv', newline='')
    csvreader = csv.reader(csvfile)
    edited_caption = []
    for row in csvreader:
        edited_caption = edited_caption + [row]
    edited_caption = edited_caption[1:]
    mapping = dict()
    for i, discrption in enumerate(edited_caption):
        if int(edited_caption[i][0]) not in mapping:
            mapping[int(edited_caption[i][0])] = list()
        mapping[int(edited_caption[i][0])] = mapping[int(edited_caption[i][0])] +\
                                             [[edited_caption[i][j] for j in \
                                              range(len(edited_caption[i])) if \
                                              edited_caption[i][j]!='#' and j>0]]
    return edited_caption, mapping
edited_caption, mapping = read_edited_caption()





max_length = len(edited_caption[0])-1
vocab_size = len(embedding_matrix)
embedding_dim = len(embedding_matrix[0])

#[train_features, mapping] = load(open("extracted_features.pickle", "rb"))




"""==========    NEW CHANGE     ==============="""

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints


import keras


import tensorflow as tf

# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 4
BUFFER_SIZE = 1000
embedding_dim = 512
units = 512
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


EPOCHS = 5
epochs = 5
number_pics_per_batch = 16
steps = len(edited_caption)//number_pics_per_batch


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
    self.Drop = tf.keras.layers.Dropout(0.5)
    self.BatchNormalization = tf.keras.layers.BatchNormalization()

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # you get 1 at the last axis because you are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * self.BatchNormalization(self.Drop(features))
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
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

    self.embedding = tf.keras.layers.Embedding(vocab_size, 300, weights=[embedding_matrix])
    #self.embedding.set_weights([embedding_matrix])
    self.embedding.trainable = False
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size,activation='softmax')
    self.Drop = tf.keras.layers.Dropout(0.5)
    self.BatchNormalization = tf.keras.layers.BatchNormalization()
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

    x = self.Drop(x)
    x = self.BatchNormalization(x)
    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))



encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


def loss_function(real, pred):
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask

  return tf.reduce_mean(loss_)



checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)

# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []




# data generator, intended to be used in a call to model.fit_generator()
def data_generator(wordtoix, max_length, batch_size):
    X, y = list(), list()
    n=0
    # loop for ever over images
    import os
    features_path = 'features\\'
    while 1:
        for i, name in enumerate(os.listdir(features_path)):
            [photos, descriptions] = load(open(features_path + name, "rb"))
            for key, desc_list in descriptions.items():
                # retrieve the photo feature
                photo = photos[key]
                for desc in desc_list:
                    n += 1
                    # encode the sequence
                    seq = [wordtoix[word] for word in desc]
                    seq = seq+[word2ix['#'] for i in range(max_length-len(seq))]

                    X.append(photo)
                    y.append(seq)
                    # yield the batch data
                    if n==batch_size:
                        yield [array(X), array(y)]
                        X, y = list(), list()
                        n=0



@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([word2ix['$']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]-1):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          out_seq = [target[j][i] for j in range(len(target))]
          loss += loss_function(out_seq, predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss




for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch,(img_tensor, target)) in enumerate(data_generator(word2ix,max_length, number_pics_per_batch)):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

        if (batch==steps):
            break


    loss_plot.append(total_loss / steps)

    ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/steps))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))






plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()



























"""#################################################################################"""
"""#################################################################################"""
"""#################################################################################"""
"""#########################        TEST       #####################################"""
"""#################################################################################"""
"""#################################################################################"""




def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    features = encoder(image)
    features = tf.expand_dims(features, 0)

    dec_input = tf.expand_dims([word2ix['$']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = np.argmax(predictions[0])
        #predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()


        if ix2word[predicted_id] == 'E':
            return result, attention_plot
        result.append(ix2word[predicted_id])

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot




import os
#image directory

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img,img_to_array
model_IV3 = InceptionV3(include_top=False,weights='imagenet')
model_IV3 = Model(inputs=model_IV3.inputs, outputs=model_IV3.layers[-1].output)



def plot_attention(image, result, attention_plot):
    temp_image = np.array(image)

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def test_picture():
    image_path=".\\test picture\\"
    features = {}
    for i, name in enumerate(os.listdir(image_path)):
        temp_name = int(name.split('.')[0])
        filename = image_path + name
        image = load_img(filename, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model_IV3.predict(image, verbose=0)
        features[name] = feature[0].reshape((-1, feature[0].shape[-1]))


    for pic in  list(features.keys()):
        image = features[pic]
        x=plt.imread(image_path+str(pic))
        plt.imshow(x)
        plt.show()
        result, attention_plot = evaluate(image)
        print("Greedy:", str(result))
        filename = image_path + pic
        image = load_img(filename, target_size=(299, 299))
        #plot_attention(image, result, attention_plot)

test_picture()







