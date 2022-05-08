"""here we pre process the raw datasset to ds_train and ds_test
    @Zejin
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import re,string


from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

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

def structured_data_preprocessing(dfdata):
    """1-1 Preprcessing structured data, take titanic survive or not  as example

    Args:
        dfdata (_type_): pandas raw data format
    """    

    dfresult= pd.DataFrame()

    #Pclass
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)



def get_dataset_from_list_files(train_path,test_path, BATCH_SIZE):
    """1-2 Preprcessing Image Data, take loading from list_files as example
    """

    def load_image(img_path,size = (32,32)):
    
        label = tf.constant(1,tf.int8) if tf.strings.regex_full_match(img_path,".*automobile.*") \
                else tf.constant(0,tf.int8)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img) #In jpeg format
        img = tf.image.resize(img,size)/255.0
        return(img,label)

    # Parallel pre-processing using num_parallel_calls and caching data with prefetch function to improve the performance
    ds_train = tf.data.Dataset.list_files(train_path) \
            .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
            .prefetch(tf.data.experimental.AUTOTUNE)  

    ds_test = tf.data.Dataset.list_files(test_path) \
            .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(BATCH_SIZE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test  

def get_txt_dataset_from_csv(train_data_path,test_data_path,MAX_WORDS = 10000, MAX_LEN = 200, BATCH_SIZE = 20 ):
    """1-3 Preprcessing txt_dataset_from_csv, take IMDB movie reviews as example
    """
    #Constructing data pipeline
    def split_line(line):
        arr = tf.strings.split(line,"\t")
        label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]),tf.int32),axis = 0)
        text = tf.expand_dims(arr[1],axis = 0)
        return (text,label)

    ds_train_raw =  tf.data.TextLineDataset(filenames = [train_data_path]) \
    .map(split_line,num_parallel_calls = tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

    ds_test_raw = tf.data.TextLineDataset(filenames = [test_data_path]) \
    .map(split_line,num_parallel_calls = tf.data.experimental.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)


    #Constructing dictionary
    def clean_text(text):
        lowercase = tf.strings.lower(text)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        cleaned_punctuation = tf.strings.regex_replace(stripped_html,
            '[%s]' % re.escape(string.punctuation),'')
        return cleaned_punctuation

    vectorize_layer = TextVectorization(
        standardize=clean_text,
        split = 'whitespace',
        max_tokens=MAX_WORDS-1, #Leave one item for the placeholder
        output_mode='int',
        output_sequence_length=MAX_LEN)

    ds_text = ds_train_raw.map(lambda text,label: text)
    vectorize_layer.adapt(ds_text)
    print(vectorize_layer.get_vocabulary()[0:100])


    #Word encoding
    ds_train = ds_train_raw.map(lambda text,label:(vectorize_layer(text),label)) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test_raw.map(lambda text,label:(vectorize_layer(text),label)) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test

def get_temporal_sequences_dataset(df,WINDOW_SIZE = 8):
    """1-4 Preprcessing temporal_sequences dataset_from_csv, take covid-19 prediction as example    
    """

    dfdata = df.set_index("date")
    dfdiff = dfdata.diff(periods=1).dropna()
    dfdiff = dfdiff.reset_index("date")

    dfdiff = dfdiff.drop("date",axis = 1).astype("float32")

    #Use the data of an eight-day window priorier of the date we are investigating as input for prediction
    def batch_dataset(dataset):
        dataset_batched = dataset.batch(WINDOW_SIZE,drop_remainder=True)
        return dataset_batched

    ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values,dtype = tf.float32)) \
    .window(WINDOW_SIZE,shift=1).flat_map(batch_dataset)

    ds_label = tf.data.Dataset.from_tensor_slices(
        tf.constant(dfdiff.values[WINDOW_SIZE:],dtype = tf.float32))
    
    #We put all data into one batch for better efficiency since the data volume is small.
    ds_train = tf.data.Dataset.zip((ds_data,ds_label)).batch(38).cache()

    return ds_train, dfdiff