import torch
from torch.utils.data import  Dataset # wraps an iterable around the dataset
import pandas as pd
import os

class ECGDataSet(Dataset):
    def __init__(self, split='train', parameter='hr'):

        self.split = split

        # data loading
        current_directory = os.getcwd()
        self.parent_directory = os.path.dirname(current_directory)
        train_small_path = os.path.join(self.parent_directory, 'data', 'deepfake-ecg-small', str(self.split) + '.csv')
        # Skip the header row
        self.df = pd.read_csv(train_small_path)  
        
        if parameter == 'hr':
            # Avg RR interval
            # in milli seconds
            RR = torch.tensor(self.df['avgrrinterval'].values, dtype=torch.float32)
            # calculate HR
            self.y = 60 * 1000/RR
        else:
            self.y = torch.tensor(self.df[parameter].values, dtype=torch.float32)
        
        # Size of the dataset
        self.samples = self.df.shape[0]

    def __getitem__(self, index):
        
        # file path
        filename= self.df['patid'].values[index]
        asc_path = os.path.join(self.parent_directory,  'data', 'deepfake-ecg-small', str(self.split), str(filename) + '.asc')
        # asc_path = os.path.join(os.getcwd(),  'data', 'deepfake-ecg-small', str(self.split), str(filename) + '.asc')
        
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
    