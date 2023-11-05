import pandas as pd
import os
import torch
import wfdb
import ecg_plot

current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
# print(current_directory)
parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
# print(parent_directory)
super_parent_directory =os.path.dirname(parent_directory)   # # /e17-4yp-Comp...
# print(super_parent_directory)

file_index = int('12035')
folder_name = str(file_index // 1000).zfill(2)+'000' 
file_name = str(file_index).zfill(5)+'_hr'

record_path = os.path.join(super_parent_directory,  'data', 'ptb-xl', 'records500', folder_name, file_name)
# print(record_path) 

# Use wfdb.rdsamp to read both the .dat file and .hea header file
record_data, record_header = wfdb.rdsamp(record_path)

print("\nRecord data shape = " + str(record_data.shape))

print("\n----------Record header----------")
print(record_header)

print("\n----------Record data----------")
print(record_data)

ecg_signals = torch.tensor(record_data) # convert dataframe values to tensor

print("\n----------ECG Tensor----------")
print(ecg_signals)
        
ecg_signals = ecg_signals.float()

print("\n----------ECG Float----------")
print(ecg_signals)

ecg_signals = ecg_signals.t()
# print("\nECG signal shape = " + str(ecg_signals.shape) + "\n") 

# ecg_plot.plot(ecg_signals, sample_rate = 500, title = 'ECG 12')
# ecg_plot.show()

# Transposing the ecg signals
ecg_signals = ecg_signals/6 # normalization
# ecg_signals = ecg_signals.t() 

print("\n----------ECG Tensor after Normalization----------")
print(ecg_signals)

print("\nECG signal shape = " + str(ecg_signals.shape) + "\n")
# print("\nECG signal shape[0] = " + str(ecg_signals[0].shape) + "\n")

# ecg_plot.plot(ecg_signals, sample_rate = 500, title = 'ECG 12')
# ecg_plot.show()


