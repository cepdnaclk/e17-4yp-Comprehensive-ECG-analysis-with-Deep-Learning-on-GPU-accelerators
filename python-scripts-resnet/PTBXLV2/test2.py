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

asc_path = os.path.join(super_parent_directory,  'data', 'deepfake-ecg-small', 'train',  '0' + '.asc')
        
ecg_signals = pd.read_csv( asc_path, header=None, sep=" ") # read into dataframe
ecg_signals = torch.tensor(ecg_signals.values) # convert dataframe values to tensor
        
ecg_signals = ecg_signals.float()

ecg_signals = ecg_signals.t()
# print("\nECG signal shape = " + str(ecg_signals.shape) + "\n") 

print(ecg_signals)

# ecg_plot.plot(ecg_signals, sample_rate = 500, title = 'ECG 12')
# ecg_plot.show()
        
# Transposing the ecg signals
ecg_signals = ecg_signals/6000 # normalization
# ecg_signals = ecg_signals.t()

print("\n----------ECG Tensor after Normalization----------")
print(ecg_signals)
print("\nECG signal shape = " + str(ecg_signals.shape) + "\n") 

# ecg_plot.plot(ecg_signals, sample_rate = 500, title = 'ECG 12')
# ecg_plot.show()
