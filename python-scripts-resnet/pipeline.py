from ECGDataSet import ECGDataSet 
from KanResWide_X2 import KanResWide_X2
from utils import checkpoint, resume, train, validate
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import wandb
import gc
import os

gc.collect()
torch.cuda.empty_cache()


# Get cpu, gpu or mps device for training 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)



# Set API Key
os.environ["WANDB_API_KEY"] = "cf61e02cee13abdd3d8a232d29df527bd6cc7f89"
# Set the WANDB_NOTEBOOK_NAME environment variable to the name of your notebook (manually)
# os.environ["WANDB_NOTEBOOK_NAME"] = "DataLoader.ipynb"
# set the WANDB_TEMP environment variable to a directory where we have write permissions
os.environ["WANDB_TEMP"] = os.getcwd()
os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()

#wandb.init(project='ECG-analysis-with-Deep-Learning-on-GPU-accelerators')




# ECG datasets
train_dataset = ECGDataSet(split='train')
validate_dataset = ECGDataSet(split='validate')

# data loaders
# It allows you to efficiently load and iterate over batches of data during the training or evaluation process.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2)
validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=2, shuffle=True, num_workers=2)





# hyperparameters
num_classes = 1  # Number of output classes
num_epochs = 20
learning_rate = 1e-3

input_shape = (8,5000)  # Modify this according to your input shape
# 128 is the batch size, 8 is the number of channels, 5000 is the number of time steps
output_size = 1  # Number of output units




model = KanResWide_X2(input_shape, output_size)
model.to(device)

# Loss function for linear values (e.g., regression)
# Mean Squared Error loss
loss_fn = nn.MSELoss()  

# Adam optimizer
# You can adjust lr and other hyperparameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  





train_losses = []
val_losses = []
epochs = []

start_epoch = 0
if start_epoch > 0:
    resume_epoch = start_epoch - 1
    resume(model, f"epoch-{resume_epoch}.pth")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    epochs.append(epoch)

    # train
    train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    train_losses.append(train_loss)

    # validation
    val_loss = validate(validate_dataloader, model, loss_fn, optimizer, device)
    val_losses.append(val_loss)

    # wandb.log({"ResNet: loss [mean absolute error] vs epoch" : wandb.plot.line_series(
    #                    xs=epochs, 
    #                    ys=[train_losses, val_losses],
    #                    keys=["training", "validation"],
    #                    title="",
    #                    xname="epochs")})
    
    #checkpoint(model, f"epoch-{epoch}.pth")

print("Done!")
# finish
# wandb.finish()

