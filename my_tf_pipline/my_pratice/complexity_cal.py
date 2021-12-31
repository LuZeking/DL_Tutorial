"""
Here we Calculate complexity
"""

import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow import keras
import matplotlib.pyplot as plt

from utils.dataset_loader import load_cifar10
from backbone.cnn import CNN_3L, Sequential_CNN_3L


# 1&2 dataset and dataloader
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
# instantiating a model from a input tensor and all imtermidate out put tensor 
layer_outputs = [layer.output for layer in model.layers[:]]
activation_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)

# img_tensor = np.random.rand(224,224,3) * 255 # tf.tensor(np.random((224,224,3)))
# from PIL import Image
# img_tensor = Image.open("/home/hpczeji1/hpc-work/Codebase/Datasets/healthy_aging/marleen_dataset/faces/17.jpg")
# img_tensor = img_tensor.resize((224,224))
# from PIL import Image
# img_tensor = Image.fromarray(img_tensor.astype('uint8')).convert('RGBA')
cat_dict = {"faces":1000,"animals":2000,"places":3000,"objects":4000}
complexity_list = []
for cat_name in cat_dict.keys():
    print(f"Caculating {cat_name} ...")
    source_dir = f"/home/hpczeji1/hpc-work/Codebase/Datasets/healthy_aging/marleen_dataset/{cat_name}/"
    batch_size = 1000
    imgs_batch_tensor = np.empty((batch_size,224,224,3))    
    for i in range(batch_size):
        img_tensor = Image.open(source_dir+f"{i+cat_dict[cat_name]-1000}.jpg")
        img_tensor = img_tensor.convert('RGB')
        # img_tensor = img_tensor.resize((224,224))
        # print(f"shape:{img_tensor.size}")
        try:
            imgs_batch_tensor[i,:,:,:] = img_tensor 
        except:
            print(f"img_tensor.size:{img_tensor.size}")
            raise NotImplementedError

    activations = activation_model.predict(imgs_batch_tensor)
    forth_layer_activation = activations[14] # 第四个 print(first_layer_activation.shape) 第三个是-13
    complexity_list += [np.mean(forth_layer_activation[i, :, :, :]) for i in range(1000)] 

# write to csv
import csv
with open("/home/hpczeji1/hpc-work/Codebase/Datasets/healthy_aging/marleen_plot/Id_Complexity.csv", "w") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Id","Complexity"])

    for i in range(4000):
        writer.writerow([i, round(complexity_list[i],2)])

# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/


# plot hist
import matplotlib.pyplot as plt

cat_dict = {"faces":1000,"animals":2000,"places":3000,"objects":4000}
cat_name = "faces"
for cat_name in cat_dict.keys():
    n, bins, patches = plt.hist(x = complexity_list[cat_dict[cat_name]-1000:cat_dict[cat_name]], bins =list(range(1,101,3)),color='#0504aa',alpha=0.5, rwidth=0.85)  # bins = [round(0.001*i,2) for i in list(range(901,1021,10))]

    # print([0.01*i for i in list(range(0,100,2))])
    plt.ylim(0,150)
    plt.xlim(0,100)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{cat_name.capitalize()} Complexity Distribution')

    save_dir = "/home/hpczeji1/hpc-work/Codebase/Datasets/healthy_aging/marleen_plot/categories/"
    plt.savefig(save_dir + f'{cat_name.capitalize()}_Complexity_Distribution.jpg', bbox_inches = 'tight')
    plt.show()
    plt.close()