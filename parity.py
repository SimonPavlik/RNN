# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:36:10 2015

Experiments with RNN applied to the parity problem.

@author: Simon
"""

import numpy as np

steps_train = 5
steps_test = 10
batchsize = 10
epoch_size = 10

# random binary input sequences
X_train = np.random.choice(2, [epoch_size, steps_train])
X_test = np.random.choice(2, steps_test)

# output parity sequences from binary input sequences
Y_train = np.cumsum(X_train, axis=1) % 2
Y_test = np.cumsum(X_test) % 2

# sequence -> 3D (# batchsize, # steps_train, # features)
X_train3D = np.empty((epoch_size, steps_train, 1))
Y_train3D = np.empty((epoch_size, steps_train, 1))

X_train3D[:, :, 0] = X_train
Y_train3D[:, :, 0] = Y_train

X_test3D = np.empty((1, steps_test, 1))
Y_test3D = np.empty((1, steps_test, 1))

X_test3D[0, :, 0] = X_test
Y_test3D[0, :, 0] = Y_test


import theano.tensor as t


def steeper_sigmoid(x):
    slope = 10
    sigmoid = 1/(1+t.exp(-slope*x))
    return sigmoid

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN
'''
RNNlayer_weights = [np.array([[1, 1]]),
                    np.array([[1,  1], [-1,  -1]]),
                    np.array([-0.5,  -1.5])]
'''
RNNlayer_weights = None
'''
output_weights = [np.array([[1], [-1]]),
                  np.array([-0.5])]
'''
output_weights = None

model = Sequential()

model.add(SimpleRNN(input_dim=1, output_dim=2, init='normal',
                    inner_init='orthogonal', activation=steeper_sigmoid,
                    weights=RNNlayer_weights,
                    return_sequences=True))

model.add(Dense(2, 1, init='normal', activation=steeper_sigmoid, weights=output_weights))

model.compile(loss='binary_crossentropy', optimizer='Adagrad')

initialWeights = model.get_weights()

history = model.fit(X_train3D, Y_train3D, batch_size=batchsize, nb_epoch=1000, show_accuracy=True)

score = model.evaluate(X_test3D, Y_test3D, show_accuracy=True)

print("score (loss, accuracy):")
print(score)

print("predicted output:")
print(model.predict(X_test3D, verbose=1))
print("actual output:")
print(Y_test)
print("actual input:")
print(X_test)
print('initial weights:')
print(initialWeights)
print('trained weights:')
print(model.get_weights())
# model.save_weights('vahy', overwrite=True)
