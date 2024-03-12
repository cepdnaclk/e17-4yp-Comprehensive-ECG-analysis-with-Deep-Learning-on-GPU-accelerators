from KanResWide_X2 import KanResWide_X2
from PTBXLV2.ECGDataSet_PTB_XL import ECGDataSet_PTB_XL
from ConvolutionalResNet import ConvolutionalResNet
from torch.utils.data import DataLoader
import torchvision
import torch
from utils import validate
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from earlystopping import EarlyStopping
import os

# 128 is the batch size, 8 is the number of channels, 5000 is the number of time steps
input_shape = (8, 5000)  
# Number of output units
output_size = 1 
# number of epochs
number_of_epochs = 1
#
learning_rate = 0.0005
#
y_parameters = ['pr', 'qrs', 'qt', 'hr']


device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )



for y_parameter in y_parameters:

    # model
    model = KanResWide_X2(input_shape, output_size)
    

    # check point path
    checkpoint_dir_path = "/storage/projects2/e17-4yp-compreh-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/checkpoints"
    checkpoint_path = checkpoint_dir_path + '/' + y_parameter + '_best_.pt'

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    print("Model loaded successfully with weights")

    # ECG dataset
    train_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='train')
    print("Train dataset loaded successfully")
    #exit()
    validate_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='validate')
    print("Validate dataset loaded successfully")
    validate_notscaled_dataset = ECGDataSet_PTB_XL(parameter=y_parameter, split='validate', scale=False)
    print("Dataset loaded successfully")

    # data loaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=20)
    validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=16, shuffle=False, num_workers=20)
    validate_notscaled_dataloader = DataLoader(dataset=validate_notscaled_dataset, batch_size=16, shuffle=False, num_workers=20)
    print("Dataloaders loaded successfully")


    patience = 50
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    # Create a StepLR scheduler that reduces the learning rate by a factor of 0.5 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # Loss function for linear values (e.g., regression)
    loss_fn = nn.MSELoss()  # Mean Squared Error loss

    # validate
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Create a SummaryWriter instance
    # writer = SummaryWriter()

    # for epoch in range(number_of_epochs):
        # validation
        # val_loss = validate(validate_dataloader, model, loss_fn, device)

        # writer.add_scalars(f"Transfer Learning Mean Absolute Loss vs Epoch [Y: {y_parameter}, Learning Rate: {learning_rate}]", {'Validation Loss':val_loss}, epoch)
        # print(f"parameter: {y_parameter} Validation Loss': {val_loss}")

        # early_stopping(val_loss, model, y_parameter)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

print(asc_path = os.path.join(self.parent_directory,  'data', 'deepfake_ecg_full_train_validation_test', 'test', '1234.asc'))
    
    # Close the writer
    # writer.close()

        
        












