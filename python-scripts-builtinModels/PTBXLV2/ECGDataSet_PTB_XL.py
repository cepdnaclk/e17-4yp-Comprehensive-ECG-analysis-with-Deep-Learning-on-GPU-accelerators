import torch
from torch.utils.data import  Dataset # wraps an iterable around the dataset
import pandas as pd
import os
import numpy as np
import wfdb

class ECGDataSet_PTB_XL(Dataset):

    def __init__(self, parameter='hr', split="train", scale = True):

        self.pr_min_val = 0
        self.pr_max_val = 0
        self.qt_min_val = 0
        self.qt_max_val = 0
        self.qrs_min_val = 0
        self.qrs_max_val = 0

        self.df = pd.read_csv('PTBXLpreprocessed.csv')
        # data loading
        current_directory = os.getcwd()                             # /e17-4yp-Comp.../python-scripts-resnet/PTB-XL
        parent_directory = os.path.dirname(current_directory)       # /e17-4yp-Comp.../python-scripts-resnet
        self.super_parent_directory = parent_directory   # # /e17-4yp-Comp...


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
        #print("train df done")
        validation = self.df.iloc[validation_indices]
        test = self.df.iloc[test_indices]

        # Reset the index for each DataFrame
        train.reset_index(drop=True, inplace=True)
        #print the size of the train dataframe
        #print("Train size: ", train.shape[0])
        #print(train.head())
        #exit()
        validation.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        #print("Before train")

        if (parameter == 'pr'):
            train = train.dropna(subset=['PR_Int_Global']) 
            column = train['PR_Int_Global']
            self.pr_min_val = column.min()
            self.pr_max_val = column.max()

            # print the values
            #print(self.pr_min_val)
            #print(self.pr_max_val)
            #exit()
        
        elif (parameter == 'qt'):
            train = train.dropna(subset=['QT_Int_Global']) 
            column = train['QT_Int_Global']
            #print(column)
            self.qt_min_val = column.min()
            self.qt_max_val = column.max()
            #print(self.qt_min_val)
            #print(self.qt_max_val)
            #exit()
            
        elif (parameter == 'qrs'):
            train = train.dropna(subset=['QRS_Dur_Global'])
            column = train['QRS_Dur_Global']
            self.qrs_min_val = column.min()
            self.qrs_max_val = column.max()

        
        if split=="train":
            self.df = train
        if split=="validate":
            self.df = validation
        elif split=="test":
            self.df = test

        #print("After train")

        if (split == 'train'):
            if parameter == 'hr':
                self.df = self.df.dropna(subset=['RR_Mean_Global'])
                # Avg RR interval
                # in milli seconds
                RR = torch.tensor(self.df['RR_Mean_Global'].values, dtype=torch.float32) 
                # calculate HR
                self.y = 60 * 1000/RR
            elif parameter == 'qrs':
                self.df = self.df.dropna(subset=['QRS_Dur_Global']) 
                column = self.df['QRS_Dur_Global']
                min_value = column.min()
                max_value = column.max()
                #print(min_value)
                #print(max_value)
                scaled_column = (column - min_value) / (max_value - min_value)
                self.y = torch.tensor(scaled_column.values, dtype=torch.float32)

            elif parameter == 'qt':
                self.df = self.df.dropna(subset=['QT_Int_Global']) 
                column = self.df['QT_Int_Global']
                min_value = column.min()
                max_value = column.max()
                #print(min_value)
                #print(max_value)
                scaled_column = (column - min_value) / (max_value - min_value)
                self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
        
            elif parameter == 'pr': 
                self.df = self.df.dropna(subset=['PR_Int_Global'])
                column = self.df['PR_Int_Global']
                min_value = column.min()
                max_value = column.max()
                # print the values
                #print(train)
                #print(min_value)
                #print(max_value)
                scaled_column = (column - min_value) / (max_value - min_value)
                self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
                #print the size of y
                #print(self.y.shape)
                #exit()

        else:
            if(scale):
                if parameter == 'hr':
                    self.df = self.df.dropna(subset=['RR_Mean_Global'])
                    #column = self.df
                    # Avg RR interval
                    # in milli seconds
                    RR = torch.tensor(self.df['RR_Mean_Global'].values, dtype=torch.float32) 
                    # calculate HR
                    self.y = 60 * 1000/RR
                elif parameter == 'qrs':
                    self.df = self.df.dropna(subset=['QRS_Dur_Global']) 
                    column = self.df['QRS_Dur_Global']
                    scaled_column = (column - self.qrs_min_val) / (self.qrs_max_val - self.qrs_min_val)
                    self.y = torch.tensor(scaled_column.values, dtype=torch.float32)

                elif parameter == 'qt':
                    self.df = self.df.dropna(subset=['QT_Int_Global']) 
                    column = self.df['QT_Int_Global']
                    scaled_column = (column - self.qt_min_val) / (self.qt_max_val - self.qt_min_val)
                    self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
        
                elif parameter == 'pr': 
                    self.df = self.df.dropna(subset=['PR_Int_Global'])
                    column = self.df['PR_Int_Global']
                    scaled_column = (column - self.pr_min_val) / (self.pr_max_val - self.pr_min_val)
                    self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
            else:
                if parameter == 'hr':   
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

    def getMinVals(self):
        return self.pr_min_val, self.qt_min_val, self.qrs_min_val
    
    def getMaxVals(self):
        return self.pr_max_val, self.qt_max_val, self.qrs_max_val

    def __getitem__(self, index):
        
        # file path
        file_index = int(self.df['ecg_id'].values[index])
        folder_name = str(file_index // 1000).zfill(2)+'000' 
        file_name = str(file_index).zfill(5)+'_hr'

        #print(file_name)

        #ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl', 'records500', folder_name, file_name)
        ecg_record_path = os.path.join(self.super_parent_directory,  'data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1', 'records500', folder_name, file_name)

        # Use wfdb.rdsamp to read both the .dat file and .hea header file
        ecg_record_data, ecg_record_header = wfdb.rdsamp(ecg_record_path)

        # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] - 12 original channels
        # ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] - 8 channels as deepfake data set

        columns_to_remove = [2, 3, 4, 5]    # remove channels - 'III', 'AVR', 'AVL', 'AVF'
        ecg_record_data = np.delete(ecg_record_data, columns_to_remove, axis=1)

        ecg_signals = torch.tensor(ecg_record_data) # convert dataframe values to tensor
        #print(ecg_signals)
        #print("*********************")
        ecg_signals = ecg_signals.float()
        
        # Transposing the ecg signals
        ecg_signals = ecg_signals/6   # normalization
        #print(ecg_signals)
        #exit()
        ecg_signals = ecg_signals.t() 
        
        qt = self.y[index]
       # print(qt)
       # exit()
        # Retrieve a sample from x and y based on the index
        return ecg_signals, qt

    def __len__(self):
        # Return the total number of samples in the dataset
        return self.samples