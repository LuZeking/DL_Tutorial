"""
Here we visulize filters and feature maps
"""

import tensorflow as tf
import numpy as np

from tensorflow import keras

from utils.dataset_loader import load_cifar10
from backbone.cnn import CNN_3L, Sequential_CNN_3L
import matplotlib.pyplot as plt



# 1&2 dataset and dataloader
dataloader,(x_train, y_train), (x_test, y_test)  =  load_cifar10(BATCH_SIZE = 64, buffer_size=1024)

# 3 model 
model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
# model.load_weight(xxx) # load pretrain, so no need for # 4&5 set optimizer and train
model.summary()
print(model.layers)
print(len(model.layers))
# [<tensorflow.python.keras.engine.input_layer.InputLayer object at 0x2b8adb5ae588>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8a9da00e80>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8adb5a8ef0>, <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x2b8ae489c668>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae489ce48>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae48d49b0>, <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x2b8ae48d47b8>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae48df908>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae48e9780>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae48e97f0>, <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x2b8ae48ef780>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae48effd0>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae48f6dd8>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae4901160>, <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x2b8ae4901ac8>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae490a5f8>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae4914470>, <tensorflow.python.keras.layers.convolutional.Conv2D object at 0x2b8ae4914898>, <tensorflow.python.keras.layers.pooling.MaxPooling2D object at 0x2b8ae491a470>, <tensorflow.python.keras.layers.core.Flatten object at 0x2b8ae491acf8>, <tensorflow.python.keras.layers.core.Dense object at 0x2b8ae49257b8>, <tensorflow.python.keras.layers.core.Dense object at 0x2b8ae4925a90>, <tensorflow.python.keras.layers.core.Dense object at 0x2b8ae4925fd0>]
# len = 23

# instantiating a model from a input tensor and all imtermidate out put tensor 
layer_outputs = [layer.output for layer in model.layers[:]]
activation_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)

img_tensor = np.random.rand(224,224,3) * 255 # tf.tensor(np.random((224,224,3)))
from PIL import Image
img_tensor = Image.open("3004.jpg")
# from PIL import Image
# img_tensor = Image.fromarray(img_tensor.astype('uint8')).convert('RGBA')
batch_size = 1
imgs_batch_tensor = np.empty((batch_size,224,224,3))
imgs_batch_tensor[0,:,:,:] = img_tensor 

activations = activation_model.predict(imgs_batch_tensor)
first_layer_activation = activations[-9] # 第四个 print(first_layer_activation.shape)

print(f"size : {first_layer_activation.shape}") #! why activation value is so huge than 256
print(f"Img 0 : mean energy value in layer 4: {np.mean(first_layer_activation[0, :, :, :]), np.max(first_layer_activation[0, :, :, 1]),np.min(first_layer_activation[0, :, :, :]),np.max(first_layer_activation[0, :, :, :])}")
plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis') # channel 4

plt.savefig("first_layer_activation.jpg")

#! 1 above plot activations, and 2 follow later link can plot filters(weight units) 
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/