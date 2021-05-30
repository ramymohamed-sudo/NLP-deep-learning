


# from __future__ import print_function, division
# from builtins import range
#  sudo pip3 install -U future

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
# from pos_baseline import get_data
# from sklearn.utils import shuffle
# from util import init_weight 
# from datetime import datetime
# from sklearn.metrics import f1_score

# from tensorflow.contrib.rnn import static_rnn as get_rnn_output
# from tensorflow.contrib.rnn import BasicRNNCell, GRUCell

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D, Embedding, LSTM

# unlike theano, tf require all the sequences to have the same length
# data is N X T X D
# - N samples, each sample of length T (might be with padded zeros),
# and D is the word vector dimensionality
# allow to process data in batches


# The first column contains the current word, the second its part-of-speech tag 
# as derived by the Brill tagger and the third its chunk tag 
# as derived from the WSJ corpus. The chunk tags contain the 
# name of the chunk type, for example I-NP for noun phrase words and
# I-VP for verb phrase words. Most chunk types have two types of chunk tags,
# B-CHUNK for the first word of the chunk and I-CHUNK for each other word in the chunk. 


def get_data(split_sequences=False):
    if not os.path.exists('chunking'):
        print('please create a folder called chunking')
        print('please put train.txt and test.txt there')
        print('please check the comments to get the download link')

    word2idx = {}
    tag2idx = {}
    word_idx = 1
    tag_idx = 1
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []


    for line in open('chunking/train.txt'):
        line = line.rstrip()

        if line:
            r = line.split()        # [word, pos, chunk tag]
            word, tag, _ = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])

            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])

        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY


    ####### load and score test data
    Xtest = []
    Ytest = []
    currentX = []
    currentY = []

    for line in open('chunking/test.txt'):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx)   # use this as unknown
            currentY.append(tag2idx[tag])

        elif split_sequences:
            Xtest.append(currentX)
            Ytest.append(currentY)
            # print("currentX", currentX, "len(currentX)", len(currentX))
            # print("currentY", currentY, "len(currentY)", len(currentY))

            currentX = []
            currentY = []

    if not split_sequences:
        Xtest = currentX
        Ytest = currentY

    return Xtrain, Ytrain, Xtest, Ytest, word2idx



# our data is list of list
def flatten(l):
    return [item for sublist in l for item in sublist]


# get the data
Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
V = len(word2idx) + 2  # vocab_size + (+1 for unknown +1 b/c start from 1)
K = len(set(flatten(Ytrain)) | set(flatten(Ytest))) + 1  # num_classes # maybe + 1 classes for the unknown class
print("V", V, "K ", K)
# print("Xtrain[0]", Xtrain[0], len(Xtrain[0]))
# print("Ytrain[0]", Ytrain[0], len(Ytrain[0]))
# print("type(Xtrain) and type(Ytrain)", type(Xtrain) and type(Ytrain)) returns List


# pad sequences
sequence_length = max(len(x) for x in Xtrain + Xtest)
print("sequence_length", sequence_length)

Xtrain = pad_sequences(Xtrain, maxlen=sequence_length, padding='post')
Ytrain = pad_sequences(Ytrain, maxlen=sequence_length, padding='post')
Xtest = pad_sequences(Xtest, maxlen=sequence_length, padding='post')
Ytest = pad_sequences(Ytest, maxlen=sequence_length, padding='post')
print('Xtrain.shape: ', Xtrain.shape)
print('Ytrain.shape: ', Ytrain.shape)


# training config
epochs = 10
learning_rate = 1e-2
mu = 0.99
batch_size = 32
hidden_layer_size = 20
embedding_dim = 10

model = Sequential()
model.add(Embedding(input_dim=V, output_dim=embedding_dim, input_length=sequence_length))
model.add(LSTM(hidden_layer_size, return_sequences=True))
# return_sequences=False gives error "ValueError: Shapes (None, 78, 45) and (None, 45) are incompatible"
model.add(Dense(K))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
                )
print(model.summary())


# model = Sequential()
# model.add(Embedding(input_dim=V, output_dim=embedding_dim, input_length=sequence_length))
# model.add(Bidirectional(LSTM(hidden_layer_size, return_sequences=True)))
# model.add(Dense(K))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', 
#                 optimizer='adam', 
#                 metrics=['accuracy']
#                 )
# print(model.summary())



def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


cat_train_tags_y = to_categorical(Ytrain, K)
print("cat_train_tags_y[0]: ", cat_train_tags_y[0])
print("......")
print("cat_train_tags_y.shape: ", cat_train_tags_y.shape)

model.fit(Xtrain, to_categorical(Ytrain, K), batch_size=128, epochs=5, validation_split=0.2)
scores = model.evaluate(Xtest, to_categorical(Ytest, K))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")


# # fit the model
# model.fit(Xtrain, Ytrain, epochs=5, batch_size=32, verbose=1)
# # evaluate the model
# loss, accuracy = model.evaluate(Xtrain, Ytrain, verbose=1)
# print(f'Accuracy = {accuracy} and loss = {loss}')




# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# print("Training Model .....")
# r = model.fit(data_train,y_train,epochs=10,validation_data=(data_test,y_test))
# o,h,c = model.predict(X)where h,c are the final hidden states of the LSTM


# # inputs
# inputs = pass
# targets = pass
# num_samples = pass
# # GRU cell - activation = relu - num_units = hidden_layer_size
# # embedding - GRU - Dense
