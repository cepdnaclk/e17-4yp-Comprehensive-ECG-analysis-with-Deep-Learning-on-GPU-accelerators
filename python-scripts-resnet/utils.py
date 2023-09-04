# def MAE(losses):
#     error_sum = 0
#     for loss in losses:
#         absolute_error = abs(loss - 0)  # Assuming 0 is the target value
#         error_sum += absolute_error

#     mean_absolute_error = error_sum / len(losses)
#     return mean_absolute_error


import numpy as np
import torch
 
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    

    train_losses_epoch = [] 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #print(X.shape)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_losses_epoch.append(loss.item())
    
    return np.mean(train_losses_epoch)


def validate(dataloader, model, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    val_losses_epoch = []

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute predictions
            pred = model(X)
            loss = loss_fn(pred, y)

            val_losses_epoch.append(loss.item())

    return np.mean(val_losses_epoch)