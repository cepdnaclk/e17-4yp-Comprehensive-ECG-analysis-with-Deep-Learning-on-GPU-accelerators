from torch import nn

class PatchEmbed(nn.Module):
    """Split image (ECG in our case) into patches and then embed them.

    ECG --> 8,5000

    Paramerters
    ----------
    img_size : int
        Size of image (ECG) in pixels (samples).    (This is 1D 5000)

    patch_size : int

    in_chans : int
        Number of input channels. (This is 8)

    embed_dim : int
        Embedding dimension.

    Attributes
    ----------

    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding.

    """
    # This class is modified so that it works with 1D data.
    def __init__(self, img_size=5000, patch_size=50, in_chans=8, embed_dim=768):
        super().__init__()
        img_size = img_size
        patch_size = patch_size
        self.n_patches = (img_size // patch_size)

        # embed_dim is the output channel size of the convolutional layer.
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape is `(batch_size, in_chans, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape is `(batch_size, n_patches, embed_dim)`.

        """

        x = self.proj(x) # (batch_size, embed_dim, n_patches)
        # I dont think flatten is needed for 1D data.
        #x = x.flatten(2) # flatten with 1st 2 dims intact
        # (batch_size, embed_dim, n_patches) --> (batch_size, n_patches, embed_dim)
        x = x.transpose(1,2)    # (batch_size, n_patches, embed_dim)
        # print the shape of x
        #print(x.shape)

        return x