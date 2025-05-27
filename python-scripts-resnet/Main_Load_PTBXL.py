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
import pandas as pd


# 128 is the batch size, 8 is the number of channels, 5000 is the number of time steps
input_shape = (8, 5000)  
# Number of output units
output_size = 1 
# number of epochs
number_of_epochs = 1000
#
# lr = 0.0005
learning_rate = 1.0
#
#y_parameters = ['hr', 'pr', 'qt', 'qrs']

# specify the parameters after a shutdown
#y_parameters = [ 'pr', 'qt', 'qrs', 'hr']

y_parameters = [ 'qt', 'qrs', 'hr']
#y_parameters = ['pr']

for y_parameter in y_parameters:

    # model
    #print("Model Start")
    model = KanResWide_X2(input_shape, output_size)

    # check point path
    checkpoint_dir_path = "/storage/projects2/e17-4yp-compreh-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/cp_test/PTBXL"
    checkpoint_path = checkpoint_dir_path + '/' + y_parameter + '_best_.pt'

    # Load the checkpoint
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully with weights")
    #print(model)
    #print("Model loaded successfully with weights")
    #num_features = model.fc.in_features     #32
    #print(num_features)
    #exit()
    #for param in model.parameters():
        #param.requires_grad = False

    # # Make the parameters of the last layer trainable
    # model.fc.weight.requires_grad = True
    # model.fc.bias.requires_grad = True
    #model.fc = nn.Linear(num_features, 1)
    #print(model)
   # print("Model loaded successfully with weights")

    # ECG dataset
    #train_dataset = ECGDataSet(parameter=y_parameter, split='train',scale=False)
    #validate_dataset = ECGDataSet(parameter=y_parameter, split='validate',scale=False)
    #validate_notscaled_dataset = ECGDataSet(parameter=y_parameter, split='validate', scale=False)

    #print("Datasets Done")

    # batch size = 16 
    # data loaders
    #train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=32)
    #validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=64, shuffle=False, num_workers=32)
    #validate_notscaled_dataloader = DataLoader(dataset=validate_notscaled_dataset, batch_size=64, shuffle=False, num_workers=32)

    #print("Dataloaders Done")
    # train and validate
    #resnet = ConvolutionalResNet(model, learning_rate, number_of_epochs, y_parameter)
    #print(f"-------------{y_parameter}---------------")
    #resnet.train_and_validate(train_dataloader, validate_dataloader, validate_notscaled_dataloader, y_parameter)
    #print("\n")
    # ECG dataset
    train_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='train',scale=False)
    print("train dataset loaded successfully")
    # get the shape of the training dataset
    #print(train_dataset[0][0].shape)
    validate_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='validate',scale = False)
    print("validate dataset loaded successfully")
    validate_notscaled_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='validate', scale=False)
    print("Datasets loaded successfully")

    
    # data loaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=32)
    #get the shape of the train dataloader
    #print(train_dataloader.dataset[0][0].shape)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=128, shuffle=False, num_workers=32)
    validate_notscaled_dataloader = DataLoader(dataset=validate_notscaled_dataset, batch_size=128, shuffle=False, num_workers=32)
    print("Dataloaders loaded successfully")

    #exit()

    #print("Dataloaders Done")
    # train and validate
    resnet = ConvolutionalResNet(model, learning_rate, number_of_epochs, y_parameter)
    print(f"-------------{y_parameter}---------------")
    resnet.train_and_validate(train_dataloader, validate_dataloader, validate_notscaled_dataloader, y_parameter)
    print("\n")