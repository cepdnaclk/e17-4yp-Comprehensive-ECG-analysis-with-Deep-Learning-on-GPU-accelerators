import datetime
from ECGDataSet import ECGDataSet 
from KanResWide_X2 import KanResWide_X2
from utils import checkpoint, resume, train, validate
from ConvolutionalResNet import ConvolutionalResNet
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import wandb
import gc
import os


# Get today's date
today = datetime.date.today()
# Create the directory path
directory_path = os.path.join("./figures", today.strftime("%Y-%m-%d"))
# Create the directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)





# 128 is the batch size, 8 is the number of channels, 5000 is the number of time steps
input_shape = (8,5000)  # Modify this according to your input shape
# Number of output units
output_size = 1 
# number of epochs
number_of_epochs = 100
#
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
#
y_parameters = ['pr', 'qrs', 'qt', 'hr']

# model
model = KanResWide_X2(input_shape, output_size)


for y_parameter in y_parameters:

    # ECG dataset
    train_dataset = ECGDataSet(split='train')
    validate_dataset = ECGDataSet(split='validate')

    # data loaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=20)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=128, shuffle=False, num_workers=20)

    for learning_rate in learning_rates:
        # train and validate
        resnet = ConvolutionalResNet(model, learning_rate, number_of_epochs, y_parameter ,directory_path + f"/{y_parameter}_{learning_rate}.png")
        resnet.train_and_validate(train_dataloader, validate_dataloader)
        
        












