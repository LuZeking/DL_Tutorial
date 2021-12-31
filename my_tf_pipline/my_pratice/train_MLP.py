import os
import sys
import tensorflow as tf
import numpy as np

from tensorflow import keras

import pandas as pd
from d2l import tensorflow as d2l
import csv

sys.path.append("/home/hpczeji1/hpc-work/Codebase/AllAbout_DeepLearning/Tutorial/my_tf_pipline/")
from utils.dataset import download,download_all,download_extract,get_kaggle_house_dataset
from utils.dataset_loader import load_cifar10,  id_table_dataset_loader
from backbone.cnn import CNN_3L, Sequential_CNN_3L
from backbone.mlp import Stacked_Regression_MLP, Sequential_Regression_MLP,get_Sequential_Regression_MLP

# 1&2 dataset load and data preprocess
train_data, test_data = get_kaggle_house_dataset()
print(f"train&test shape: {train_data.shape, test_data.shape}")
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]) # visualize the first4 and last 2 features
train_features, test_features, train_labels, n_train = id_table_dataset_loader(train_data, test_data)
# dataloader,(x_train, y_train), (x_test, y_test)  =  load_cifar10(BATCH_SIZE = 64, buffer_size=1024)

# # 3 model 

model = get_Sequential_Regression_MLP() # Sequential_Regression_MLP() #  Sequential_CNN_3L(input_shape = (32,32,3)) 

# model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

# # 4&5 set optimizer and train
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer="Adam", loss = tf.keras.losses.mean_squared_logarithmic_error) # ,metrics=['accuracy'] # Kers have log version of MeanSquaredError 

# reshaped_x_train = x_train.reshape(50000, 32, 32, 3).astype("float32") / 255
model.fit(train_features , train_labels, epochs=200,batch_size=32,verbose=0)
model.summary() #! K-fold traning can also be used to calculate better loss, in case of overfitting

print("Start to predict...")
predictions = model.predict(test_features)
# model.evaluate() # when have labels, you can use evalute # use print(dir(tf.random/tf.model)) to see sub func
# test_imgs_list = test_data.iloc[:,1] #! if shuffled, how to get testset' name order?
# assert len(test_imgs_list) == len(predictions)
# with open("/home/hpczeji1/hpc-work/Codebase/AllAbout_DeepLearning/Tutorial/my_tf_pipline/results/results/mlp_predictions_submit.csv") as f:
#     writer = csv.writer(f,dialect=",")
#     # for i in range(len(test_imgs_list)):
#     writer.writerows([test_imgs_list,predictions])

 #! save in pandas way is more suffient, and meet the kaggle requirements
predictions = np.array(predictions)
test_data["SalePrice"] = pd.Series(predictions.reshape(1,-1)[0]) #! same with (-1,) or (-1,1) 
# print(predictions.shape)
# print(predictions.reshape(1,-1))
submission = pd.concat([test_data["Id"], test_data["SalePrice"]],axis=1)  # test data from pd.read_csv
submission.to_csv("../results/submission.csv", index = False)