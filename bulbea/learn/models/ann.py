from __future__ import absolute_import
from six import with_metaclass

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers import core

from bulbea.learn.models import Supervised

class ANN(Supervised):
    pass

class RNNCell(object):
    RNN  = recurrent.SimpleRNN
    GRU  = recurrent.GRU
    LSTM = recurrent.LSTM

class RNN(ANN):
    def __init__(self, sizes,
                 cell       = RNNCell.LSTM,
                 dropout    = 0.2,
                 activation = 'linear',
                 loss       = 'mse',
                 optimizer  = 'rmsprop',
                 metrics    = ["mae"]):
        self.model = Sequential()
        self.model.add(cell(
            units            = sizes[1],
            return_sequences = True
        ))

        for i in range(2, len(sizes) - 1):
            self.model.add(cell(units     = sizes[i],
                                #input_dim = sizes[i-1],
                                #output_dim = sizes[i],
                                return_sequences = False))
            self.model.add(core.Dropout(dropout))

        self.model.add(core.Dense(units = sizes[-1]))
        self.model.add(core.Activation(activation))

        self.model.compile(loss = loss, 
                           optimizer = optimizer, 
                           metrics = metrics)

    def fit(self, X, y, *args, **kwargs):
        return self.model.fit(X, y, *args, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath, *args, **kwargs):
        return self.model.save(filepath)

    def close(self):
        K.clear_session()
        del self.model
