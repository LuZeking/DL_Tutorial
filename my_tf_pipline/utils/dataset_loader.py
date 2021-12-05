import tensorflow as tf
import numpy as np

from tensorflow import keras

def load_cifar10(BATCH_SIZE = 64, buffer_size=1024):
    (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.cifar10.load_data()
    print(f"CIFAR10 x_train shape:{x_train.shape}, y_train shape : {y_train.shape}")
    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train.reshape(50000, 32, 32, 3).astype("float32") / 255, y_train)  # not reshape ot 1024*3
    )
    BATCH_SIZE = 64
    dataloader = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

    return dataloader,  (x_train, y_train), (x_test, y_test) 