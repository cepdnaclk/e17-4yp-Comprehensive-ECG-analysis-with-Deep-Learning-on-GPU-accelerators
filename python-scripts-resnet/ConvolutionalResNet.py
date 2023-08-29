import torch
from utils import train, validate
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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

        self.model = model.to(self.device)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.fig_path = fig_path
        self.predict = predict

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
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
            writer.add_scalar(f"Train Loss vs Epoch [Y: {self.predict}, Learning Rate: {self.learning_rate}]", train_loss, epoch)
            writer.add_scalar(f"Validation Loss vs Epoch [Y: {self.predict}, Learning Rate: {self.learning_rate}]", val_loss, epoch)

        # Close the writer
        writer.close()


    '''
    To start trainning and validation
    '''
    def train_and_validate(self, train_dataloader, validate_dataloader):
        
        train_losses = []
        val_losses = []
        epochs = []
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            epochs.append(epoch)

            # train
            train_loss = train(train_dataloader, self.model, self.loss_fn, self.optimizer, self.device)
            train_losses.append(train_loss)

            # validation
            val_loss = validate(validate_dataloader, self.model, self.loss_fn, self.device)
            val_losses.append(val_loss)

        # plot graph
        self.plot_graph(epochs, train_losses, val_losses)

        
        
