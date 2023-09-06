import torch
from utils import train, validate
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from earlystopping import EarlyStopping

class ConvolutionalResNet():

    '''
    Constructor
    '''
    def __init__(self, model, learning_rate, epochs, predict, fig_path):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # early stopping patience; how long to wait after last time validation loss improved.
        self.patience = 10

        self.model = model.to(self.device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.fig_path = fig_path
        self.predict = predict

        self.optimizer = optim.NAdam(model.parameters(), lr=learning_rate)

        # Create a StepLR scheduler that reduces the learning rate by a factor of 0.5 every 10 epochs
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Loss function for linear values (e.g., regression)
        self.loss_fn = nn.MSELoss()  # Mean Squared Error loss


    def plot_graph(self, epochs, train_losses, val_losses):

        # # Create a new figure
        # plt.figure()

        # # Plot the first line
        # plt.plot(epochs, train_losses, label='training', color='blue')

        # # Plot the second line
        # plt.plot(epochs, val_losses, label='validation', color='red')

        # # Add labels and title
        # plt.xlabel('epochs')
        # plt.ylabel('losses')
        # plt.title(f"Mean Absolute Loss vs Epoch [Y: {self.predict}, Learning Rate: {self.learning_rate}]")
        # plt.legend()  # Add legend based on label names

        # # Show the plot
        # plt.show()
        # # save figure
        # plt.savefig(self.fig_path)

        # Create a SummaryWriter instance
        writer = SummaryWriter()

        # Log training and validation losses
        for epoch, train_loss, val_loss in zip(epochs, train_losses, val_losses):
            writer.add_scalars(f"Mean Absolute Loss vs Epoch [Y: {self.predict}, Learning Rate: {self.learning_rate}]", {'Training Loss':train_loss, 'Validation Loss':val_loss}, epoch)

        # Close the writer
        writer.close()


    '''
    To start trainning and validation
    '''
    def train_and_validate(self, train_dataloader, validate_dataloader, y_parameter):
        
        train_losses = []
        val_losses = []
        epochs = []
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            epochs.append(epoch)

            # train
            train_loss = train(train_dataloader, self.model, self.loss_fn, self.optimizer, self.device)
            train_losses.append(train_loss)

            # validation
            val_loss = validate(validate_dataloader, self.model, self.loss_fn, self.device)
            val_losses.append(val_loss)

            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(val_loss, self.model, y_parameter)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            # after every epoch version most common
            self.scheduler.step()  # decay LR (if step_size hit)

        # plot graph
        self.plot_graph(epochs, train_losses, val_losses)

        

        
        
