import numpy as np
import tensorflow as tf

from tensorflow import keras

from utils.dataset_loader import load_cifar10
from backbone.cnn import CNN_3L, Sequential_CNN_3L

# 1&2 dataset and dataloader
dataloader,(x_train, y_train), (x_test, y_test)  =  load_cifar10(BATCH_SIZE = 64, buffer_size=1024)

# 3 model 
model =  CNN_3L() #  Sequential_CNN_3L(input_shape = (32,32,3)) 
model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# 4&5 set optimizer and train
model.compile(optimizer="SGD", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),metrics=['accuracy'])

reshaped_x_train = x_train.reshape(50000, 32, 32, 3).astype("float32") / 255
model.fit(reshaped_x_train , y_train, epochs=1)
model.summary()