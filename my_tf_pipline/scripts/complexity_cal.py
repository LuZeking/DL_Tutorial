"""
@Zejin Lu
Here we Calculate visual complexity by calculating the mean value of layer 4 output in VGG16
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
import matplotlib.pyplot as plt


# build odel 
model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
model.summary()
print(model.layers)
print(len(model.layers))


# instantiating a model from a input tensor and all imtermidate output tensor 
layer_outputs = [layer.output for layer in model.layers[:]]
activation_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)


cat_dict = {"faces":1000,"animals":2000,"places":3000,"objects":4000}
layer_activation_dict = {1:3,2:6,3:10,4:14}
for layer in layer_activation_dict.keys(): 
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
        forth_layer_activation = activations[layer_activation_dict[layer]] # 第四个 print(first_layer_activation.shape) 第三个是-13
        complexity_list += [np.mean(forth_layer_activation[i, :, :, :]) for i in range(1000)] 

    # write to csv
    import csv
    with open(f"/home/hpczeji1/hpc-work/Codebase/Datasets/healthy_aging/marleen_plot/Layer{layer}_Id_Complexity.csv", "w") as f:
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
        plt.savefig(save_dir + f'Layer{layer}_{cat_name.capitalize()}_Complexity_Distribution.jpg', bbox_inches = 'tight')
        plt.show()
        plt.close()


    # plot togethear
    import pandas as pd
    csv_file = f"/home/hpczeji1/hpc-work/Codebase/Datasets/healthy_aging/marleen_plot//Layer{layer}_Id_Complexity.csv"
    df = pd.read_csv(csv_file)

    normalized_df=(df-df.min())/(df.max()-df.min())*100 # norm to 0-100
    complexity_list = normalized_df["Complexity"]
    cat_dict = {"faces":1000,"animals":2000,"places":3000,"objects":4000}

    legend = list(cat_dict.keys())
    xlabel, ylabel = "Complexity Value", "Count"
    hist_data_list = [complexity_list[:1000], complexity_list[1000:2000], complexity_list[2000:3000], complexity_list[3000:]]
    title = "Complexity Distribution across Categories"

    import sys
    sys.path.append("/home/hpczeji1/hpc-work/Codebase/AllAbout_DeepLearning/plot_tools/")
    import plot_basic as plot
    plot.show_multi_hist(legend, xlabel, ylabel, hist_data_list, title = title, savefig = True, save_path= save_dir+f"layer_{layer}_",bins =40)

    cat_dict = {"faces":1000,"animals":2000,"places":3000,"objects":4000}

    for cat_name in cat_dict.keys():
        hist_data_list = complexity_list[cat_dict[cat_name]-1000:cat_dict[cat_name]]
        print(f"{cat_name, [np.min(hist_data_list), np.max(hist_data_list)], round(np.mean(hist_data_list),2), round(np.median(hist_data_list),2)}")


    import csv
    with open(save_dir+f"layer_{layer}_range_mean_median.csv", "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["Categories","Range","Mean", "Median"])
        for cat_name in cat_dict.keys():
            hist_data_list = complexity_list[cat_dict[cat_name]-1000:cat_dict[cat_name]]
            writer.writerow([cat_name, [round(np.min(hist_data_list),2), round(np.max(hist_data_list),2)], round(np.mean(hist_data_list),2), round(np.median(hist_data_list),2)])