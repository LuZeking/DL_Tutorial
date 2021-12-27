import tensorflow as tf
import numpy as np

from tensorflow import keras
 

""" 3 way to build models in tensofrlow 2.X 
    
   
    1 normal model building
    2 sequential funcition
    3 stacked layers
"""
class Hypercolum_layer(keras.layers.Layer):
    """一个把activity稀疏正则化加入损失函数的Layer。"""

    def __init__(self, filters = 64, kernel_size=(3,3), activation=tf.nn.relu):
        super(Hypercolum_layer, self).__init__()
        self.LCN = keras.layers.LocallyConnected2D(filters, kernel_size=kernel_size, activation=activation) 

    def call(self, inputs):
        # add spatial loss to nomral LCN
        x = self.LCN(inputs)

        # 1 reshape to 2d sheet 
        position_num,kernel_size_with_c,weight_channels = self.LCN.kernel.shape
        sheet_2D_kernel_LCN_1 = tf.reshape(self.LCN.kernel, (30*3*8, 30*9*8))
        

        # #2 spatial loss equation to  implement sheet
        top_left_LCN_1 = sheet_2D_kernel_LCN_1[:-1,:-1]
        bottom_right_LCN_1 = sheet_2D_kernel_LCN_1[1:, 1:]
        cosine_similarity_LCN_1 = tf.multiply(top_left_LCN_1, bottom_right_LCN_1)/(tf.norm(top_left_LCN_1)*tf.norm(bottom_right_LCN_1))
        loss_hypercolum = tf.reduce_mean((1-cosine_similarity_LCN_1)/2)
        # loss_hypercolum = tf.reduce_mean(sheet_2D_kernel_LCN_1)

        alpha = 0.1
        self.add_loss(alpha*loss_hypercolum)  #! without lambda, if this loss iterm will be optimize? Need to deeper understand the  mechanism of Update
        # self.add_loss(lambda: alpha*loss_hypercolum)  # self.add_loss(tf.reduce_sum(inputs))

        return x

class CNN_3L(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.conv_1 = keras.layers.Conv2D(64,  kernel_size=(3,3), activation=tf.nn.relu) # 
        self.lcn_1 = Hypercolum_layer(64,  kernel_size=(3,3), activation=tf.nn.relu)
        self.flatten = keras.layers.Flatten(data_format='channels_last')
        self.linear = keras.layers.Dense(10)
    
    def call(self, inputs):
        x = self.lcn_1(inputs)
        x = self.flatten(x)
        return self.linear(x)

def Sequential_CNN_3L(image_shape = (32,32,3)):
    """2 sequential funcition
    """
    inputs = tf.keras.Input(shape=image_shape)  # ,batch_size=64
    # d = tf.keras.layers.Dense(10)
    # LCN_1 = keras.layers.Conv2D(64,  kernel_size=(3,3), activation=tf.nn.relu)
    LCN_1 =  keras.layers.LocallyConnected2D(64, kernel_size=(3,3), activation=tf.nn.relu)
    x = LCN_1(inputs)
    Flatten_1 = keras.layers.Flatten(data_format='channels_last')
    x = Flatten_1(x)
    outputs = tf.keras.layers.Dense(10)(x)


    model = tf.keras.Model(inputs, outputs)



    # Weight regularization.
    print(LCN_1.kernel.shape)  # LCN weight shape (900, 27, 64)(position_num,kernel3*3*3,channels), while CNN weight shape is  (3, 3, 3, 64)(image_channel, kernelsize1&2,weight channels)
    #!current reshape way may not be right, better the weight be square
    # 1 reshape to 2d sheet 
    position_num,kernel_size_with_c,weight_channels = LCN_1.kernel.shape
    sheet_2D_kernel_LCN_1 = tf.reshape(LCN_1.kernel, (30*3*8, 30*9*8)) #  ((position_num**0.5)*kernel_size_with_c*weight_channels, (position_num**0.5)*kernel_size_with_c*weight_channels)
    # sheet_2D_kernel_LCN_1 = LCN_1.kernel.reshape((position_num**0.5)*kernel_size_with_c*weight_channels, (position_num**0.5)*kernel_size_with_c*weight_channels)  # (30*27*64, 30*27*64)
    print(f"after reshape to 2D sheet:{sheet_2D_kernel_LCN_1.shape}")

    #2 spatial loss equation to  implement sheet
    top_left_LCN_1 = sheet_2D_kernel_LCN_1[:-1,:-1]
    bottom_right_LCN_1 = sheet_2D_kernel_LCN_1[1:, 1:]
    cosine_similarity_LCN_1 = np.dot(top_left_LCN_1, tf.transpose(bottom_right_LCN_1))/(np.linalg.norm(top_left_LCN_1)*np.linalg.norm(bottom_right_LCN_1))
    loss_hypercolum = tf.reduce_mean((1-cosine_similarity_LCN_1)/2)
    print(f"loss_hypercolum:{loss_hypercolum}")

    alpha = 0.1
    model.add_loss(lambda: alpha*loss_hypercolum)  # tf.reduce_mean(sheet_2D_kernel_LCN_1)

    return model


class Stacked_CNN_3L(keras.layers.Layer):
    """3 stacked layers: 3 layers CNN."""

    def __init__(self):
        super(CNN_3L, self).__init__() 
        self.conv_1 = keras.layers.Conv2D(64,  kernel_size=(3,3), activation=tf.nn.relu) # 
        self.flatten = keras.layers.Flatten(data_format='channels_last')
        self.linear = keras.layers.Dense(10)

    def call(self, inputs):
        x = self.conv_1(inputs)
        # x = tf.nn.relu(x)
        x = self.flatten(x)
        # x = tf.nn.relu(x)
        # self.add_loss(tf.reduce_sum(inputs))  #TODO 修改loss即可完成Spatial Loss的定义
        return self.linear(x)


"""Sequential way 2 to build vgg16"""
