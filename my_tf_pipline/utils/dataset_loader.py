"""here we pre process the datasset"""
import tensorflow as tf
import numpy as np
import pandas as pd

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


def id_table_dataset_loader(train_data, test_data):
    """Preprocess the id-table dataset like kaggle house price prediction"""

    all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:])) # remove useless ID info, and concat train and test

    # 2.1 standrize data for better optimization and avoid unbalance coefficient
    numeric_features = all_features.dtypes[all_features.dtypes!="object"].index  # object means not float or int, and .index return pd name_list
    # print(f"all_features.dtype : {all_features.dtypes} \n {numeric_features}")
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean())/x.std())
    # after standarizing, we don't need to use mean anymore, so we can replace NAN with 0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 2.2--One-hot encoding(from str features/Discrete values to numeric)
    all_features = pd.get_dummies(all_features, dummy_na=True) # for dummy_na=True,nan" (missing value) is considered a valid feature value and an indicator feature is created for it
    # print(f"all_features.shape: {all_features.shape}") # afer one-hot encoding, num of features from 79 to 331

    # 2.3 train&test&label setting
    n_train = train_data.shape[0]
    train_features = tf.constant(all_features[:n_train].values, dtype = tf.float32)  # use .valuse to get value from 2D framedata
    test_features = tf.constant(all_features[n_train:].values, dtype=tf.float32)
    train_labels = tf.constant(train_data.SalePrice.values.reshape(-1,1),dtype=tf.float32)
    assert len(train_features) == len(train_labels)
    print(f"train and test shape:{train_features.shape, test_features.shape}")

    return train_features, test_features, train_labels, n_train
