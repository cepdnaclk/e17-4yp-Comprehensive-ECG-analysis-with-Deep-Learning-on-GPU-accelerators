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
learning_rate = 0.1
#
#y_parameters = ['hr', 'pr', 'qt', 'qrs']
y_parameters = [ 'pr', 'qt', 'qrs', 'hr']
#y_parameters = ['qt'] 

'''
# data preprocessing
# data loading
current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
super_parent_directory = parent_directory   # # /e17-4yp-Comp...

features_csv_path = os.path.join(super_parent_directory,  'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'features', '12sl_features.csv')   
        # features_csv_path = os.path.join(super_parent_directory,  'data', 'ptb-xl+', 'features', '12sl_features.csv') 

statements_csv_path = os.path.join(super_parent_directory,  'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'labels', 'ptbxl_statements.csv')
        # statements_csv_path = os.path.join(super_parent_directory,  'data', 'ptb-xl+', 'labels', 'ptbxl_statements.csv')  

        # Skip the header row
df = pd.read_csv(features_csv_path) 
statements_df = pd.read_csv(statements_csv_path)

print("df loaed successfully")

        # Create an empty list to store the indices of rows to be removed
rows_to_remove = [] 

        # Create an empty list to store the indices of rows to be removed for non NORM ecg
rows_to_remove_norm = []

        # Iterate through the rows
for index, row in df.iterrows():
    #print(index)
    file_index = int(df['ecg_id'].values[index])
    folder_name = str(file_index // 1000).zfill(2)+'000' 
    file_name = str(file_index).zfill(5)+'_hr.hea'
    ecg_record_path = os.path.join(super_parent_directory,  'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1', 'records500', folder_name, file_name)
            # ecg_record_path = os.path.join(super_parent_directory,  'data', 'ptb-xl', 'records500', folder_name, file_name)
            #print(ecg_record_path)

            # Check if the ecg_record_path exists
    if not os.path.exists(ecg_record_path):
        rows_to_remove.append(index)

        # Remove rows where ecg_record_path does not exist
df.drop(rows_to_remove, inplace=True)
        # Reset the DataFrame index if needed
df.reset_index(drop=True, inplace=True)

        # Iterate through the rows to remove non normal ecg
print("Before for loop 2")
for index, row in statements_df.iterrows():
            #print(index)
    ecg_id = statements_df['ecg_id'].values[index]
    scp_codes = statements_df['scp_codes'].values[index]

    if 'NORM' not in scp_codes:
        rows_to_remove_norm.append(ecg_id)

print("Before for loop 3")
        # Removing the non normal ecg
for index, row in df.iterrows():
    print(index)
    if row['ecg_id'] in rows_to_remove_norm:
        df.drop(index, inplace=True)

        # Reset the DataFrame index 
df.reset_index(drop=True, inplace=True)

print(df.head())

df.to_csv("PTBXLpreprocessed.csv", index=False) 

exit()
'''

for y_parameter in y_parameters:

    # model
    model = KanResWide_X2(input_shape, output_size)

    # check point path
    checkpoint_dir_path = "/storage/localSSD/e17-4yp-comprehensive-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/checkpoints/DFtest"
    checkpoint_path = checkpoint_dir_path + '/' + y_parameter + '_best_.pt'

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    #print(model)
    #print("Model loaded successfully with weights")
    num_features = model.fc.in_features     #32
    #print(num_features)
    #exit()
    for param in model.parameters():
        param.requires_grad = False

    # # Make the parameters of the last layer trainable
    # model.fc.weight.requires_grad = True
    # model.fc.bias.requires_grad = True
    model.fc = nn.Linear(num_features, 1)
    #print(model)
    print("Model loaded successfully with weights")

   # exit()

    # ECG dataset
    train_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='train')
    print("train dataset loaded successfully")
    # get the shape of the training dataset
    #print(train_dataset[0][0].shape)
    validate_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='validate')
    print("validate dataset loaded successfully")
    validate_notscaled_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='validate', scale=False)
    print("Datasets loaded successfully")

    #get min values
    pr_min, qt_min, qrs_min = train_dataset.getMinVals()
    #print(pr_min, qt_min, qrs_min)
    #getmax values
    pr_max, qt_max, qrs_max = train_dataset.getMaxVals()
    #print(pr_max, qt_max, qrs_max)

    #exit()

    #print("Datasets Done")

    # batch size = 16 
    # data loaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=32)
    #get the shape of the train dataloader
    #print(train_dataloader.dataset[0][0].shape)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=1, shuffle=False, num_workers=32)
    validate_notscaled_dataloader = DataLoader(dataset=validate_notscaled_dataset, batch_size=1, shuffle=False, num_workers=32)
    print("Dataloaders loaded successfully")

    #exit()

    #print("Dataloaders Done")
    # train and validate
    resnet = ConvolutionalResNet(model, learning_rate, number_of_epochs, y_parameter)
    print(f"-------------{y_parameter}---------------")
    resnet.train_and_validate_tl(train_dataloader, validate_dataloader, validate_notscaled_dataloader, y_parameter, pr_max, pr_min, qt_max, qt_min, qrs_max, qrs_min)
    print("\n")
        










