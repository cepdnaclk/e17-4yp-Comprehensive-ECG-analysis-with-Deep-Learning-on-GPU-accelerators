import torch
from torch.utils.data import  Dataset # wraps an iterable around the dataset
import pandas as pd
import os



class ECGDataSet(Dataset):
    def __init__(self, split='train', parameter='hr', scale = True):

        self.split = split

        self.pr_min_val = 0
        self.pr_max_val = 0
        self.qt_min_val = 0
        self.qt_max_val = 0
        self.qrs_min_val = 0
        self.qrs_max_val = 0

        # data loading
        current_directory = os.getcwd()
        self.parent_directory = os.path.dirname(current_directory)
        train_small_path = os.path.join(self.parent_directory, 'data', 'deepfake_ecg_full_train_validation_test/clean', str(self.split) + '.csv')
        #train_small_path = os.path.join(self.parent_directory, 'data', 'deepfake-ecg-small', str(self.split) + '.csv')
        # Skip the header row
        self.df = pd.read_csv(train_small_path)  

        # get the min max values from train dataset
        train = os.path.join(self.parent_directory, 'data', 'deepfake_ecg_full_train_validation_test/clean', 'train' + '.csv')
        self.train_df = pd.read_csv(train)

        if (parameter == 'pr'):
            column = self.train_df[parameter]
            self.pr_min_val = column.min()
            self.pr_max_val = column.max()
        
        elif (parameter == 'qt'):
            column = self.train_df[parameter]
            self.qt_min_val = column.min()
            self.qt_max_val = column.max()
        
        elif (parameter == 'qrs'):
            column = self.train_df[parameter]
            self.qrs_min_val = column.min()
            self.qrs_max_val = column.max()

        if (split == 'train'):
            if parameter == 'hr':
                # Avg RR interval
                # in milli seconds
                RR = torch.tensor(self.df['avgrrinterval'].values, dtype=torch.float32)
                # calculate HR
                self.y = 60 * 1000/RR
            # just divided by 6000 to normalize
            else:
                #having min max scaling
                if (scale):
                    column = self.df[parameter]
                    min_value = column.min()
                    max_value = column.max()
                    scaled_column = (column - min_value) / (max_value - min_value)
                    self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
                else:
                    column = self.df[parameter]
                    self.y = torch.tensor(column.values, dtype=torch.float32)
            #print(self.y)
            #exit()
        else:
            if(scale):
                if parameter == 'hr':
                    # Avg RR interval
                    # in milli seconds
                    RR = torch.tensor(self.df['avgrrinterval'].values, dtype=torch.float32)
                    # calculate HR
                    self.y = 60 * 1000/RR
                    # just divided by 6000 to normalize
                else:
                    #having min max scaling
                    column = self.df[parameter]
                    #min_value = column.min()
                    #max_value = column.max()

                    if (parameter == 'pr'):
                        scaled_column = (column - self.pr_min_val) / (self.pr_max_val - self.pr_min_val)
                        self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
                    elif (parameter == 'qt'):
                        scaled_column = (column - self.qt_min_val) / (self.qt_max_val - self.qt_min_val)
                        self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
                    elif (parameter == 'qrs'):
                        scaled_column = (column - self.qrs_min_val) / (self.qrs_max_val - self.qrs_min_val)
                        self.y = torch.tensor(scaled_column.values, dtype=torch.float32)
            else:
                if parameter == 'hr':
                    # Avg RR interval
                    # in milli seconds
                    RR = torch.tensor(self.df['avgrrinterval'].values, dtype=torch.float32)
                    # calculate HR
                    self.y = 60 * 1000/RR
                else:
                    #just divided by 6000 to normalize
                    column = self.df[parameter]
                    self.y = torch.tensor(column.values, dtype=torch.float32)
                
            #print(self.y)
            #exit()
        
        
        # Size of the dataset
        self.samples = self.df.shape[0]


    def __getitem__(self, index):
        
        # file path
        filename= self.df['patid'].values[index]
        asc_path = os.path.join(self.parent_directory,  'data', 'deepfake_ecg_full_train_validation_test', str(self.split), str(filename) + '.asc')
        #asc_path = os.path.join(os.getcwd(),  'data', 'deepfake-ecg-small', str(self.split), str(filename) + '.asc')
        #asc_path = os.path.join(self.parent_directory,  'data', 'deepfake-ecg-small', str(self.split), str(filename) + '.asc')
        
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
    
