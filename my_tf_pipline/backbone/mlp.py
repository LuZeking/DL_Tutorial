import tensorflow as tf
import numpy as np

from tensorflow import keras

""" 3 way to build models in tensofrlow 2.X 
    
   
    1 normal model building
    2 sequential funcition
    3 stacked layers
"""

class Stacked_Regression_MLP(keras.layers.Layer):
    """3 stacked layers: 1 layers MLP."""

    def __init__(self):
        super(Stacked_Regression_MLP, self).__init__() 
        # self.conv_1 = keras.layers.Conv2D(64,  kernel_size=(3,3), activation=tf.nn.relu) # 
        # self.flatten = keras.layers.Flatten(data_format='channels_last')
        self.linear1 = keras.layers.Dense(100)
        self.linear2 = keras.layers.Dense(10)
        self.linear3 = keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2()) # for regression problem, the final output is only 1 value
        # use l2 regularizers and weight decay (.l2(weight_decay)) 

    def call(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        return self.linear3(x)

def Sequential_Regression_MLP():
    """WRONG 2 sequential funcition
    """
    inputs = tf.keras.Input()  # ,batch_size=64

    x = keras.layers.Dense(100)(inputs)  # no activation
    x = keras.layers.Dense(10)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)

    return model

def get_Sequential_Regression_MLP():
    net = keras.models.Sequential()
    net.add(keras.layers.Dense(1))
    return net