import torch.nn.functional as F
from torch import nn

class KanResInit(nn.Module):
    def __init__(self, in_channels, filterno_1, filterno_2, filtersize_1, filtersize_2, stride):
        #print(in_channels) --> 8
        super(KanResInit, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filterno_1, filtersize_1, stride=stride)
        self.bn1 = nn.BatchNorm1d(filterno_1)
        self.conv2 = nn.Conv1d(filterno_1, filterno_2, filtersize_2)
        self.bn2 = nn.BatchNorm1d(filterno_2)
        # initialize a relu layer
        #self.relu1 = nn.ReLU()
        #self.relu2 = nn.ReLU()
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        #print(x.shape)
        #print(x)
        x = self.conv1(x)
        #print(x.shape)
        #print(x)
        #exit()
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


