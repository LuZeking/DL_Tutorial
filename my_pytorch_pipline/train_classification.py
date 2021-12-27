import torch 
import os 
import torchvision

from torch.utils.data import Dataset, DataLoader
from utils.dataset import load_Minst

from backbone.cnn import SimpleModel
from torch import nn
from torch import optim
from tqdm import tqdm
from datetime import datetime

# 1&2 Dataset and Dataloader
train_dataset, test_dataset = load_Minst(dataset_path = "/home/hpczeji1/hpc-work/Codebase/Datasets/mnist_data")
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=10000,
                          shuffle=True)

# 3 Model 
model = SimpleModel()
mode = "test" # "train"

if mode == "train":
    # 4&5 optimzer setting and train
    optimizer = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    epoch_num = 2
    for epoch in range(epoch_num):
        with tqdm(train_loader) as train_bar:
            for x,y in train_bar:
                optimizer.zero_grad()
                loss = loss_fn(model(x),y)
                loss.backward()
                optimizer.step()
            print(f"epoch:{epoch}, loss{loss}")


    time = str(datetime.now()).split(" ")[0].replace("-", "_")
    torch.save(model.state_dict(), f"simple_model_{time}.pth")

elif mode == "test":
    model.load_state_dict(torch.load("./model_saved/simple_model_2021_12_06.pth")) 
    print(model(test_dataset) #! didn't match because some of the arguments have invalid types: (MNIST, Parameter, Parameter, tuple, tuple, tuple, int)

else:
    raise NotImplementedError


print("~~~~~~撒花~~~~~~")
