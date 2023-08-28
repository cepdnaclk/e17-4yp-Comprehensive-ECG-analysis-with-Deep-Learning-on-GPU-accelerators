# def MAE(losses):
#     error_sum = 0
#     for loss in losses:
#         absolute_error = abs(loss - 0)  # Assuming 0 is the target value
#         error_sum += absolute_error

#     mean_absolute_error = error_sum / len(losses)
#     return mean_absolute_error


import numpy as np
import torch
 
# def checkpoint(model, filename):
#     torch.save(model.state_dict(), filename)
    
# def resume(model, filename):
#     model.load_state_dict(torch.load(filename))

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
    
    return np.mean(np.abs(train_losses_epoch))


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

    return np.mean(np.abs(val_losses_epoch))


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'checkpoints/best_model.pth')


def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'checkpoints/final_model.pth')