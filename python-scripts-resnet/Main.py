import datetime
from ECGDataSet import ECGDataSet 
from KanResWide_X2 import KanResWide_X2
from PTBXLV2.ECGDataSet_PTB_XL import ECGDataSet_PTB_XL
from utils import train, validate
from ConvolutionalResNet import ConvolutionalResNet
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import wandb
import gc
import os


# 128 is the batch size, 8 is the number of channels, 5000 is the number of time steps
input_shape = (12, 5000)  # Modify this according to your input shape // change to (12,5000) for ptbxl
# Number of output units
output_size = 1 
# number of epochs
number_of_epochs = 500
#
learning_rate = 0.0005
#
y_parameters = ['pr', 'qrs', 'qt', 'hr']


for y_parameter in y_parameters:

    # ECG dataset
    train_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='train')
    validate_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='validate')

    # data loaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=20)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=16, shuffle=False, num_workers=20)

    # model
    model = KanResWide_X2(input_shape, output_size)

    # train and validate
    resnet = ConvolutionalResNet(model, learning_rate, number_of_epochs, y_parameter)
    resnet.train_and_validate(train_dataloader, validate_dataloader, y_parameter)
        
        












