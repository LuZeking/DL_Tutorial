import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import backend as f

from d2l import tensorflow as d2l

#@save
class RNNModel(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        # rnn返回两个以上的值
        Y, *state = self.rnn(X, state)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)

def get_Sequential_Regression_LSTM(units = 64, output_size = 1):
    
    # lstm_layer = keras.layers.RNN(
    #         keras.layers.LSTMCell(units)) # , input_shape=(None, input_dim)
    
    num_hiddens = 256
    run_cell = tf.keras.layers.SimpleRNNCell(num_hiddens, kernel_initializer = "glorot_uniform") # DEFAULT
    rnn_layer = tf.keras.layers.RNN(
        run_cell,
        time_major=True,  # If True, the inputs and outputs will be in shape `(timesteps, batch, ...)`, otherwise reverse
        return_sequences=True,
        return_state=True)
    net = RNNModel(rnn_layer,vocab_size=output_size)
    # net = keras.models.Sequential(
    #     [
    #         rnn_layer, #lstm_layer,
    #         keras.layers.BatchNormalization(),
    #         keras.layers.Dense(output_size),
    #     ]
    # )
    return net