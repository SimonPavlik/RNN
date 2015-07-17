# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:36:10 2015

Experiments with RNN applied to the parity problem.

@author: Simon
"""

import numpy as np

steps_train = 10000;
steps_test = 20
batchsize = 1000
timestamps = steps_train/batchsize
#random binary input sequences
X_train = np.random.choice(2,[batchsize, timestamps])
X_test = np.random.choice(2,steps_test)

#output parity sequences from binary input sequences
Y_train = np.cumsum(X_train, axis=0)%2
Y_test = np.cumsum(X_test)%2

#sequence -> 3D (# batchsize, # timesteps, # features)
X_train3D = np.empty((batchsize,timestamps, 1)) 
Y_train3D = np.empty((batchsize,timestamps, 1))

X_train3D[:,:,0] = X_train
Y_train3D[:,:,0] = Y_train

X_test3D = np.empty((1,steps_test, 1)) 
Y_test3D = np.empty((1,steps_test, 1))

X_test3D[0,:,0] = X_test
Y_test3D[0,:,0] = Y_test


import theano.tensor as T

def steeper_sigmoid(x):
    slope = 5
    sigmoid = 1/(1+T.exp(-slope*x))    
    return sigmoid

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import SimpleRNN

RNNlayer_weights = [np.array([[ 1  , 1]]), np.array([[1 ,  -1],
                     [ 1,  -1]]), np.array([ 1,  -1])]
#RNNlayer_weights = None


model = Sequential()

model.add(SimpleRNN(input_dim=1, output_dim=2, init='glorot_uniform',
                                 inner_init='orthogonal', activation=steeper_sigmoid,
                                 weights=RNNlayer_weights, truncate_gradient=-1,
                                 return_sequences=True))

model.add(Dense(2, 1, activation=steeper_sigmoid))

model.compile(loss='binary_crossentropy', optimizer='RMSprop')

model.fit(X_train3D, Y_train3D, batch_size=batchsize, nb_epoch=20)
score = model.evaluate(X_test3D, Y_test3D, show_accuracy=True)

print("score (loss, accuracy):")
print(score)

print("predicted output:")
print(model.predict(X_test3D, verbose=1))
print("actual output:")
print(Y_test)
print("actual input:")
print(X_test)
print('weights:')
print(model.get_weights())