from torch import nn

class MLP(nn.Module):
    """Multi-layer perceptron.
    
    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The first linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape is `(batch_size, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape is `(batch_size, n_patches + 1, out_features)`.

        """

        x = self.fc1(x) # (batch_size, n_patches + 1, hidden_features)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)    # (batch_size, n_patches + 1, out_features)

        return x
