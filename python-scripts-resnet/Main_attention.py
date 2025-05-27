import datetime
from ECGDataSet import ECGDataSet 
from KanResWide_X2_attention import KanResWide_X2_attention
#from PTBXLV2.ECGDataSet_PTB_XL import ECGDataSet_PTB_XL
from PTBXLV1.ECGDataSet_PTB_XL import ECGDataSet_PTB_XL
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
input_shape = (8, 5000)  
# Number of output units
output_size = 1 
# number of epochs
number_of_epochs = 1000
#
# lr = 0.0005
learning_rate = 0.01
#
#y_parameters = ['hr', 'pr', 'qt', 'qrs']
y_parameters = [ 'pr', 'qt', 'qrs', 'hr']
#y_parameters = ['pr']

for y_parameter in y_parameters:

    # model
    #print("Model Start")
    model = KanResWide_X2_attention(input_shape, output_size)

    # ECG dataset
    train_dataset = ECGDataSet(parameter=y_parameter, split='train',scale=False)
    validate_dataset = ECGDataSet(parameter=y_parameter, split='validate',scale=False)
    validate_notscaled_dataset = ECGDataSet(parameter=y_parameter, split='validate', scale=False)

    #print("Datasets Done")

    # batch size = 16 
    # data loaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=32)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=256, shuffle=False, num_workers=32)
    validate_notscaled_dataloader = DataLoader(dataset=validate_notscaled_dataset, batch_size=256, shuffle=False, num_workers=32)

    #print("Dataloaders Done")
    # train and validate
    resnet = ConvolutionalResNet(model, learning_rate, number_of_epochs, y_parameter)
    print(f"-------------{y_parameter}---------------")
    resnet.train_and_validate(train_dataloader, validate_dataloader, validate_notscaled_dataloader, y_parameter)
    print("\n")
        