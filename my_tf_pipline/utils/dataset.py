"""Here we download dataset"""

import hashlib
import os
import tarfile
import zipfile
import requests
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import expand_dims

def get_classification_dataset_from_dir(directory, image_size=(224, 224), batch_size=32, shuffle=True, validation_split_ratio=None):
    """! from_dir inherit from tf.data.Dataset, can see more e.g. slices, txt,xx from
         https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/data/Dataset#methods """
    seed=369

    if validation_split_ratio:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(  # tf.keras.utils.image_dataset_from_directory in 2.7.0
        directory, labels='inferred', label_mode='int', class_names=None, 
        color_mode='rgb', batch_size=batch_size, image_size=image_size, 
        shuffle= shuffle, seed=seed, validation_split=validation_split_ratio, subset="training",
        interpolation='bilinear', follow_links=False,  ) # crop_to_aspect_ratio=False, not this in 2.3
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(  # tf.keras.utils.image_dataset_from_directory in 2.7.0
        directory, labels='inferred', label_mode='int', class_names=None, 
        color_mode='rgb', batch_size=batch_size, image_size=image_size, 
        shuffle= shuffle, seed=seed, validation_split=validation_split_ratio, subset="validation",
        interpolation='bilinear', follow_links=False,  ) # crop_to_aspect_ratio=False, not this in 2.3

        return train_ds,val_ds

    else:
        return  tf.keras.preprocessing.image_dataset_from_directory(  # tf.keras.utils.image_dataset_from_directory in 2.7.0
        directory, labels='inferred', label_mode='int', class_names=None, 
        color_mode='rgb', batch_size=batch_size, image_size=image_size, 
        shuffle= shuffle, seed=seed, validation_split=None, subset= None,
        interpolation='bilinear', follow_links=False,  ) # crop_to_aspect_ratio=False, not this in 2.3

def get_classification_dataset_imgslist_from_dir(dir_name, test_ratio = 0.2):
    """make dataset from dir

        inputs: 
            dir_name, abs path without last /
        return:
            train_data, train_label, test_data,test_label, class_id2name_dict     
    """
    class_name_list, class_id_list  = os.listdir(dir_name), list(range(len(os.listdir(dir_name))))
    class_id2name_dict = {k:v for k,v in zip(class_id_list, class_name_list)}
    train_data, train_labels, test_data, test_labels = [],[],[],[]
    from glob import glob
    for cat in class_name_list:
        imgs_list = glob(dir_name+"/"+cat+"/")
        for img in imgs_list:
            judge_dice = np.random.randn()
            if judge_dice <= test_ratio:
                test_data.append(img)
                test_labels.append(class_id2name_dict[cat])
            else:
                train_data.append(img)
                train_labels.append(class_id2name_dict[cat])

    return  train_data, train_labels, test_data, test_labels, class_id2name_dict 

def get_kaggle_house_dataset():
    """save your dataset url in Data_HUB, refer to MU Li' Tutorial"""

    DATA_HUB = dict()
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

    DATA_HUB['kaggle_house_train'] = (  #@save
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce')

    DATA_HUB['kaggle_house_test'] = (  #@save
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    train_data = pd.read_csv(download('kaggle_house_train', DATA_HUB, "/home/hpczeji1/hpc-work/Codebase/Datasets/datahub/kaggle_house"))
    test_data = pd.read_csv(download('kaggle_house_test', DATA_HUB, "/home/hpczeji1/hpc-work/Codebase/Datasets/datahub/kaggle_house"))
    return train_data,test_data

def download(name, DATA_HUB, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file from a DATA_HUB, returning the local file name"""
    assert name in DATA_HUB, f"{name} not existing in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  #@save
    """Download and extract the zip/tar file"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be decompressed'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all the files in the DATA_HUB"""
    for name in DATA_HUB:
        download(name)



def generate_function_seq_dataset(T = 1000, tau =4, if_plot = False):
    import tensorflow as tf

    time = tf.range(1,T+1,dtype=tf.float32)
    y = tf.sin(0.01 * time)+tf.random.normal([T],0,0.2)

    if if_plot:
        from d2l import tensorflow as d2l
        d2l.plot(time, [y], 'time', 'y', xlim=[1, 1000], figsize=(6, 3))

    # features = tf.Variable(np.empty((T-tau,tau))) # constant cannot be assigned
    features = tf.Variable(tf.zeros((T - tau, tau)))
    # labels = []
    for i in range(tau):
        # features[:,i] = y[i:T-tau+i]
        features[:, i].assign(y[i: T - tau + i])

        # labels.append(y[T-tau+i])
    labels = tf.reshape(y[tau:],(-1,1)) # tau is time length, T-tau is number of sequences

    return features,labels, y, time

    