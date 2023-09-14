import torch
from torch.utils.data import  Dataset # wraps an iterable around the dataset
import pandas as pd
import os
import wfdb

class ECGDataSet_PtbXl(Dataset):
    def __init__(self, parameter='hr'):

        # data loading
        current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
        parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
        self.super_parent_directory =os.path.dirname(parent_directory)   # # /e17-4yp-Comp...

        features_csv_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'features', '12sl_features.csv')   

        # Skip the header row
        self.df = pd.read_csv(features_csv_path)  
        
        if parameter == 'hr':   # 'hr' should be replaced
            # Avg RR interval
            # in milli seconds
            RR = torch.tensor(self.df['avgrrinterval'].values, dtype=torch.float32)     # 'avgrrinterval' should be replaced 
            # calculate HR
            self.y = 60 * 1000/RR
        else:
            self.y = torch.tensor(self.df[parameter].values, dtype=torch.float32)
        
        # Size of the dataset
        self.samples = self.df.shape[0]

    def __getitem__(self, index):
        
        # file path
        file_index = int(self.df['ecg_id'].values[index])
        folder_name = str(file_index // 1000).zfill(2)+'000' 
        file_name = str(file_index).zfill(5)+'_medians'

        ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'median_beats', '12sl-changed', '12sl-copy', folder_name, file_name)
        
        # Use wfdb.rdsamp to read both the .dat file and .hea header file
        ecg_record_data, ecg_record_header = wfdb.rdsamp(ecg_record_path)

        ecg_signals = torch.tensor(ecg_record_data) # convert dataframe values to tensor
        
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