import torch
from torch.utils.data import  Dataset # wraps an iterable around the dataset
import pandas as pd
import os
import numpy as np
import wfdb

class ECGDataSet_PTB_XL(Dataset):
    def __init__(self, parameter='hr', split="train"):

        # data loading
        current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
        parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
        self.super_parent_directory = parent_directory   # # /e17-4yp-Comp...

        # features_csv_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1', 'features', '12sl_features.csv')   
        features_csv_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl+', 'features', '12sl_features.csv')   

        # Skip the header row
        self.df = pd.read_csv(features_csv_path) 

        # Create an empty list to store the indices of rows to be removed
        rows_to_remove = [] 

        # Iterate through the rows
        for index, row in self.df.iterrows():
            file_index = int(self.df['ecg_id'].values[index])
            folder_name = str(file_index // 1000).zfill(2)+'000' 
            file_name = str(file_index).zfill(5)+'_hr.hea'
            # ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1', 'records500', folder_name, file_name)
            ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl', 'records500', folder_name, file_name)
            #print(ecg_record_path)

            # Check if the ecg_record_path exists
            if not os.path.exists(ecg_record_path):
                rows_to_remove.append(index)

        # Remove rows where ecg_record_path does not exist
        self.df.drop(rows_to_remove, inplace=True)
        # Reset the DataFrame index if needed
        self.df.reset_index(drop=True, inplace=True)

        # Assuming you already have your DataFrame loaded as self.df
        # Define the percentages for train, validation, and test sets
        train_percentage = 0.7
        validation_percentage = 0.15
        test_percentage = 0.15

        # Calculate the number of samples for each set
        total_samples = len(self.df)
        num_train = int(train_percentage * total_samples)
        num_validation = int(validation_percentage * total_samples)
        num_test = total_samples - num_train - num_validation

        # Create an array of indices to shuffle
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        # Split the shuffled indices into train, validation, and test sets
        train_indices = indices[:num_train]
        validation_indices = indices[num_train:num_train + num_validation]
        test_indices = indices[num_train + num_validation:]

        # Create DataFrames for each set
        train = self.df.iloc[train_indices]
        validation = self.df.iloc[validation_indices]
        test = self.df.iloc[test_indices]

        # Reset the index for each DataFrame
        train.reset_index(drop=True, inplace=True)
        validation.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)


        if split=="train":
            self.df = train
        if split=="validate":
            self.df = validation
        elif split=="test":
            self.df = test
        
        if parameter == 'hr':   # 'hr' should be replaced
            self.df = self.df.dropna(subset=['RR_Mean_Global'])
            # Avg RR interval
            # in milli seconds
            RR = torch.tensor(self.df['RR_Mean_Global'].values, dtype=torch.float32) 
            # calculate HR
            self.y = 60 * 1000/RR

        elif parameter == 'qrs':
            self.df = self.df.dropna(subset=['QRS_Dur_Global']) 
            self.y = torch.tensor(self.df['QRS_Dur_Global'].values, dtype=torch.float32)

        elif parameter == 'qt':
            self.df = self.df.dropna(subset=['QT_Int_Global']) 
            self.y = torch.tensor(self.df['QT_Int_Global'].values, dtype=torch.float32)
        
        elif parameter == 'pr': 
            self.df = self.df.dropna(subset=['PR_Int_Global'])
            self.y = torch.tensor(self.df['PR_Int_Global'].values, dtype=torch.float32)
        
        # Size of the dataset
        self.samples = self.df.shape[0]

    def __getitem__(self, index):
        
        # # file path
        # file_index = int(self.df['ecg_id'].values[index])
        # folder_name = str(file_index // 1000).zfill(2)+'000' 
        # file_name = str(file_index).zfill(5)+'_medians'

        # ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl+', 'median_beats', '12sl-changed', '12sl-copy', folder_name, file_name)

        # file path
        file_index = int(self.df['ecg_id'].values[index])
        folder_name = str(file_index // 1000).zfill(2)+'000' 
        file_name = str(file_index).zfill(5)+'_hr'

        ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl', 'records500', folder_name, file_name)
        # ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1', 'records500', folder_name, file_name)

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