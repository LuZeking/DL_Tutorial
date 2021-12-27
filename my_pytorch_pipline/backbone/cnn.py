"""3 model
"""

from torch import nn
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(3,3),stride=3) # in_channel is te image channel number
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3),stride =3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3),stride =3)
        self.relu = nn.ReLU(inplace = True)
        self.Flatten = nn.Flatten(start_dim=1, end_dim=-1) # (B,C,H,W), USE C,H,W
        self.Linear = nn.Linear(in_features=64*1*1, out_features=10,bias= False) # ins_feature num = flatten
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        # print("[before flatten] x.shape: {}".format(x.shape))  # torch.Size
        x = self.Flatten(x)
        # print("[after flatten] x.shape: {}".format(x.shape))  # torch.Size([1, 3920])
        x = self.Linear(x)

        return self.relu(x)