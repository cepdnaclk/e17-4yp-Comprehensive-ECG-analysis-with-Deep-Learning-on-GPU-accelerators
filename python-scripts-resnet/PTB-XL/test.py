import pandas as pd
import os
import torch
import wfdb

current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
# print(current_directory)
parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
# print(parent_directory)
super_parent_directory =os.path.dirname(parent_directory)   # # /e17-4yp-Comp...
# print(super_parent_directory)

record_path = os.path.join(super_parent_directory,  'data', 'ptb-xl', 'records500', '00000', '00001_hr')
# print(record_path) 

# Use wfdb.rdsamp to read both the .dat file and .hea header file
record_data, record_header = wfdb.rdsamp(record_path)

print("\nRecord data shape = " + str(record_data.shape))

print("\n----------Record data----------")
print(record_data)

ecg_signals = torch.tensor(record_data) # convert dataframe values to tensor

print("\n----------ECG Tensor----------")
print(ecg_signals)
        
ecg_signals = ecg_signals.float()

print("\n----------ECG Float----------")
print(ecg_signals)

# Transposing the ecg signals
ecg_signals = ecg_signals/6000 # normalization
ecg_signals = ecg_signals.t() 

print("\n----------ECG Tensor after Normalization----------")
print(ecg_signals)

print("\nECG signal shape = " + str(ecg_signals.shape) + "\n")

print(ecg_signals.shape[0])

# file_index = int('12035')
# folder_name = str(file_index // 1000).zfill(2)+'000' 
# file_name = str(file_index).zfill(5)+'_hr'

# ecg_record_path = os.path.join(super_parent_directory,  'data', 'ptb-xl', 'records500', folder_name, file_name)
# print("\n" + ecg_record_path + "\n")