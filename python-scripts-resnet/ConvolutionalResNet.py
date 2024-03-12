import torch
from utils import train, validate, validate_notscaled, validate_notscaled_tl
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from earlystopping import EarlyStopping

class ConvolutionalResNet():

    '''
    Constructor
    '''
    def __init__(self, model, learning_rate, epochs, predict):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # early stopping patience; how long to wait after last time validation loss improved.
        self.patience = 100

        self.model = model.to(self.device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.predict = predict

        self.optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
        #self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Create a StepLR scheduler that reduces the learning rate by a factor of 0.5 every 10 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Loss function for linear values (e.g., regression)
        self.loss_fn = nn.MSELoss()  # Mean Squared Error loss


    def plot_graph(self, epochs, train_losses, val_losses, val_real_losses=None):

        # Create a SummaryWriter instance
        writer = SummaryWriter()

        # Log training and validation losses
        for epoch, train_loss, val_loss, val_real_loss in zip(epochs, train_losses, val_losses, val_real_losses):
            print(epoch)
            print(f"TL Mean Absolute Loss vs Epoch [Y: {self.predict}, Learning Rate: {self.learning_rate}]", {'Training Loss':train_loss, 'Validation Loss':val_loss}, epoch)
            writer.add_scalars(f"Resnet: Mean Absolute Loss vs Epoch [Y: {self.predict}, Learning Rate: {self.learning_rate}]", {'Training Loss':train_loss, 'Validation Loss':val_loss}, epoch)

        # Close the writer
        writer.close()


    '''
    To start trainning and validation
    '''
    def train_and_validate(self, train_dataloader, validate_dataloader,validate_notscaled_dataloader, y_parameter):
        
        train_losses = []
        val_losses = []
        val_real_losses = []
        epochs = []
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(self.epochs):
            # print(f"Epoch {epoch+1}\n-------------------------------")
            epochs.append(epoch)

            # train
            train_loss = train(train_dataloader, self.model, self.loss_fn, self.optimizer, self.device, epoch)
            train_losses.append(train_loss)

            # validation
            #val_loss, val_real_loss = validate(validate_dataloader, self.model, self.loss_fn, self.device, y_parameter)
            val_loss = validate(validate_dataloader, self.model, self.loss_fn, self.device, y_parameter)
            #val_loss_real = validate_notscaled(validate_notscaled_dataloader, self.model, self.loss_fn, self.device, y_parameter)
            val_losses.append(val_loss)
            #val_real_losses.append(val_loss_real)

            print(f"{epoch} tarainning loss : {train_loss} validation loss : {val_loss}")

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.model, y_parameter)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # after every epoch version most common
            # self.scheduler.step()  # decay LR (if step_size hit)

        # plot graph
        # change this to plot real losses
        self.plot_graph(epochs, train_losses, val_losses, val_real_losses)

    # method for transfer learning
    def train_and_validate_tl(self, train_dataloader, validate_dataloader,validate_notscaled_dataloader, y_parameter, pr_max_val, pr_min_val, qt_max_val, qt_min_val, qrs_max_val, qrs_min_val):
        
        train_losses = []
        val_losses = []
        val_real_losses = []
        epochs = []
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(self.epochs):
            # print(f"Epoch {epoch+1}\n-------------------------------")
            epochs.append(epoch)

            # train
            train_loss = train(train_dataloader, self.model, self.loss_fn, self.optimizer, self.device)
            train_losses.append(train_loss)

            # validation
            #val_loss, val_real_loss = validate(validate_dataloader, self.model, self.loss_fn, self.device, y_parameter)
            val_loss = validate(validate_dataloader, self.model, self.loss_fn, self.device, y_parameter)
            #val_loss_real = validate_notscaled_tl(validate_notscaled_dataloader, self.model, self.loss_fn, self.device, y_parameter, pr_max_val, pr_min_val, qt_max_val, qt_min_val, qrs_max_val, qrs_min_val)
            val_losses.append(val_loss)
            #val_real_losses.append(val_loss_real)

            #print(f"{epoch} tarainning loss : {train_loss} validation loss : {val_loss} Real validation loss : {val_loss_real}")
            print(f"{epoch} tarainning loss : {train_loss} validation loss : {val_loss} ")

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.model, y_parameter)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # after every epoch version most common
            # self.scheduler.step()  # decay LR (if step_size hit)

        # plot graph
        # change this to plot real losses
        self.plot_graph(epochs, train_losses, val_losses, val_real_losses)

        

        
        
