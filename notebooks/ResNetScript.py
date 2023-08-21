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


import torch.optim as optim
import torch.nn.functional as F
# q: what is the difference between torch.nn.functional and torch.nn
# a: https://discuss.pytorch.org/t/what-is-the-difference-between-torch-nn-and-torch-nn-functional/33597/2

# Get cpu, gpu or mps device for training 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class ECGDataSet(Dataset):
    
    def __init__(self, split='train'):

        self.split = split

        # data loading
        current_directory = os.getcwd()
        self.parent_directory = os.path.dirname(current_directory)
        train_small_path = os.path.join(self.parent_directory, 'data', 'deepfake-ecg-small', str(self.split) + '.csv')
        self.df = pd.read_csv(train_small_path)  # Skip the header row
        
        # Avg RR interval
        # in milli seconds
        RR = torch.tensor(self.df['avgrrinterval'].values, dtype=torch.float32)
        # calculate HR
        self.y = 60 * 1000/RR

        # Size of the dataset
        self.samples = self.df.shape[0]

    def __getitem__(self, index):
        
        # file path
        filename= self.df['patid'].values[index]
        asc_path = os.path.join(self.parent_directory, 'data', 'deepfake-ecg-small', str(self.split), str(filename) + '.asc')
        
        ecg_signals = pd.read_csv( asc_path, header=None, sep=" ") # read into dataframe
        ecg_signals = torch.tensor(ecg_signals.values) # convert dataframe values to tensor
        
        ecg_signals = ecg_signals.float()
        
        # Transposing the ecg signals
        ecg_signals = ecg_signals/6000 # normalization
        ecg_signals = ecg_signals.t() 
        
        qt = self.y[index]
        # Retrieve a sample from x and y based on the index
        return ecg_signals, qt

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.samples
    

# ECG dataset
train_dataset = ECGDataSet(split='train')
validate_dataset = ECGDataSet(split='validate')

# data loader
# It allows you to efficiently load and iterate over batches of data during the training or evaluation process.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=20)
validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=128, shuffle=False, num_workers=20)

# q: what is num_workers?
# A: num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)

import torch.nn.functional as F

class KanResInit(nn.Module):
    def __init__(self, in_channels, filterno_1, filterno_2, filtersize_1, filtersize_2, stride):
        #print(in_channels) --> 8
        super(KanResInit, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, filterno_1, filtersize_1, stride=stride)
        self.bn1 = nn.BatchNorm1d(filterno_1)
        self.conv2 = nn.Conv1d(filterno_1, filterno_2, filtersize_2)
        self.bn2 = nn.BatchNorm1d(filterno_2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class KanResModule(nn.Module):
    def __init__(self, in_channels, filterno_1, filterno_2, filtersize_1, filtersize_2, stride):
        super(KanResModule, self).__init__()
        # have to use same padding to keep the size of the input and output the same
        # calculate the padding needed for same
        padding = (filtersize_1 - 1) // 2 + (stride - 1)
        self.conv1 = nn.Conv1d(in_channels, filterno_1, filtersize_1, stride=stride, padding='same')
        self.bn1 = nn.BatchNorm1d(filterno_1)
        self.conv2 = nn.Conv1d(filterno_1, filterno_2, filtersize_2, padding='same')
        self.bn2 = nn.BatchNorm1d(filterno_2)
        
    def forward(self, x):
        identity = x
        #print(x.shape)      
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + identity
        return x

class KanResWide_X2(nn.Module):
    def __init__(self, input_shape, output_size):
        super(KanResWide_X2, self).__init__()

        #print(input_shape[0])
        #print(input_shape[1])

        self.input_shape = input_shape
        self.output_size = output_size
        
        self.init_block = KanResInit(input_shape[0], 64, 64, 8, 3, 1)
        self.pool = nn.AvgPool1d(kernel_size=2)
        
        self.module_blocks = nn.Sequential(
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1),
            KanResModule(64, 64, 64, 50, 50, 1)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = self.init_block(x)
        print("init block trained")
        #print(x.shape)
        x = self.pool(x)
        print("pool 1 trained")
        #print(x.shape)
        x = self.module_blocks(x)
        print("module blocks trained")
        x = self.global_avg_pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #print(X.shape)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #if batch % 100 == 0:
         #   loss, current = loss.item(), (batch + 1) * len(X)
          #  print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


input_shape = (8,5000)  # Modify this according to your input shape
# 128 is the batch size, 8 is the number of channels, 5000 is the number of time steps

output_size = 1  # Number of output units

model = KanResWide_X2(input_shape, output_size)
model.to(device)
print(model)

import torch.optim as optim

# Loss function for linear values (e.g., regression)
loss_fn = nn.MSELoss()  # Mean Squared Error loss

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # You can adjust lr and other hyperparameters

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    #test(validate_dataloader, model, loss_fn)
print("Done!")