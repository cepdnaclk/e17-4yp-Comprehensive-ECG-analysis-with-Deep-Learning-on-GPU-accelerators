import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset # wraps an iterable around the dataset
from torchvision import datasets    # stores the samples and their corresponding labels
from torchvision.transforms import transforms  # transformations we can perform on our dataset
from torchvision.transforms import ToTensor
import pandas as pd
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt
from ResidualBlock import ResidualBlock

# Residual CNN model
class ResidualCNN(nn.Module):
    def __init__(self, num_classes):
        super(ResidualCNN, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(8, 16, kernel_size=2, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        # First residual block
        self.res_block1 = ResidualBlock(16, 16)
        # Second residual block
        self.res_block2 = ResidualBlock(16, 16)
        # Fully connected layer
        self.fc = nn.Linear(16 * 2500, num_classes)

    def forward(self, x):
        # Pass input through the initial convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Pass the output through the first residual block
        x = self.res_block1(x)
        # Pass the output through the second residual block
        x = self.res_block2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        # Pass the flattened output through the fully connected layer
        x = self.fc(x)
        return x