import torch
from torch.utils.data import  Dataset # wraps an iterable around the dataset
import pandas as pd
import os
import wfdb


# current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
# parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
# super_parent_directory =os.path.dirname(parent_directory)   # # /e17-4yp-Comp...
# # print(current_directory)        
# # print(parent_directory)
# # print(super_parent_directory)

# test_path = os.path.join(super_parent_directory, 'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'features', 'ecgdeli_features.csv')
# # print(test_path)

# df = pd.read_csv(test_path)  # Skip the header row
# # print(df)

# atr_path = os.path.join(super_parent_directory,  'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'fiducial_points', 'ecgdeli', '00000', '00001_points_global')
# # print(atr_path)

# annotation = wfdb.rdann(atr_path, 'atr')  # 'atr' indicates annotation file
# # print(annotation.sample)
# # print(annotation.symbol)

# ecg_signals = torch.tensor(annotation.sample) # convert dataframe values to tensor
# # print(ecg_signals)

# # Transposing the ecg signals
# ecg_signals = ecg_signals/6000 # normalization
# print(ecg_signals)
# ecg_signals = ecg_signals.t()
# print(ecg_signals) 

class ECGDataSet(Dataset):
    def __init__(self, parameter='hr'):

        # data loading
        current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
        parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
        self.super_parent_directory =os.path.dirname(parent_directory)   # # /e17-4yp-Comp...
        # print(current_directory)        
        # print(parent_directory)
        # print(super_parent_directory)

        test_path = os.path.join(self.super_parent_directory, 'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'features', 'ecgdeli_features.csv')
        # print(test_path)

        self.df = pd.read_csv(test_path)  # Skip the header row
        # print(self.df) 
        
        if parameter == 'hr':
            # Avg RR interval
            # in milli seconds
            RR = torch.tensor(self.df['avgrrinterval'].values, dtype=torch.float32)     # avgrrinterval ?
            # calculate HR
            self.y = 60 * 1000/RR
        else:
            self.y = torch.tensor(self.df[parameter].values, dtype=torch.float32)
        
        # Size of the dataset
        self.samples = self.df.shape[0]

    def __getitem__(self, index):
        
        # file path
        filename = self.df['ecg_id'].values[index]
        file_folder = int(filename) // 1000
        
        file_folder_str = str(file_folder).zfill(2)
        file_name_str = str(filename).zfill(5) 

        atr_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'fiducial_points', 'ecgdeli', file_folder_str+'000', file_name_str+'_points_global')
        # print(atr_path)

        annotation = wfdb.rdann(atr_path, 'atr')  # 'atr' indicates annotation file
        # print(annotation.sample)
        # print(annotation.symbol)

        ecg_signals = torch.tensor(annotation.sample) # convert dataframe values to tensor
        # print(ecg_signals)

        ecg_signals = ecg_signals.float()

        # Transposing the ecg signals
        ecg_signals = ecg_signals/6000 # normalization
        # print(ecg_signals)
        ecg_signals = ecg_signals.t()
        # print(ecg_signals) 
        
        
        # Transposing the ecg signals
        ecg_signals = ecg_signals/6000 # normalization
        ecg_signals = ecg_signals.t() 
        
        qt = self.y[index]
        # Retrieve a sample from x and y based on the index
        return ecg_signals, qt

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.samples