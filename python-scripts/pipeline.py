from ECGDataSet import ECGDataSet 
from ResidualCNN import ResidualCNN
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import wandb
import os

# Set API Key
os.environ["WANDB_API_KEY"] = "cf61e02cee13abdd3d8a232d29df527bd6cc7f89"
# Set the WANDB_NOTEBOOK_NAME environment variable to the name of your notebook (manually)
# os.environ["WANDB_NOTEBOOK_NAME"] = "DataLoader.ipynb"
# set the WANDB_TEMP environment variable to a directory where we have write permissions
os.environ["WANDB_TEMP"] = os.getcwd()
os.environ["WANDB_DIR"] = os.getcwd()
os.environ["WANDB_CONFIG_DIR"] = os.getcwd()
wandb.init(project='ECG-analysis-with-Deep-Learning-on-GPU-accelerators')

# hyperparameters
num_classes = 1  # Number of output classes
num_epochs = 20
learning_rate = 0.000001

# ECG dataset
train_dataset = ECGDataSet(split='train')
validate_dataset = ECGDataSet(split='validate')

# data loader
# It allows you to efficiently load and iterate over batches of data during the training or evaluation process.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
validate_dataloader = DataLoader(dataset=validate_dataset, batch_size=32, shuffle=True, num_workers=2)

# model
model = ResidualCNN(num_classes)

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
epochs = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    epochs.append(epoch)

    train_losses_epoch = [] 
    for batch_inputs, batch_labels in train_dataloader:

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        train_losses_epoch.append(int(loss))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    train_loss = np.mean(np.abs(train_losses_epoch))
    train_losses.append(train_loss)


    model.eval()
    with torch.no_grad():
        val_losses_epoch = []  # List to store validation losses for the current epoch
        for batch, (X_val, y_val) in enumerate(validate_dataloader):
            #X_val, y_val = X_val.to(device), y_val.to(device)

            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

            val_losses_epoch.append(int(val_loss))

        val_loss = np.mean(np.abs(val_losses_epoch))
        val_losses.append(val_loss)

    wandb.log({"ResNet: loss [mean absolute error] vs epoch" : wandb.plot.line_series(
                       xs=epochs, 
                       ys=[train_losses, val_losses],
                       keys=["training", "validation"],
                       title="",
                       xname="epochs")})

print("Done!")
# finish
wandb.finish()

