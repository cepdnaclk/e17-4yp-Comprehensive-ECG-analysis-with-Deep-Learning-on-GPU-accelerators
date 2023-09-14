from torch import nn

from Attention import Attention
from MLP import MLP

class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------

    dim : int
        Number of input channels (embed_dim).

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module relative to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------

    norm1, norm2 : nn.LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.
    
    mlp : MLP
        MLP module.

    """

    def __init__(self,dim,n_heads, mlp_ratio=4.0, qkv_basis=True, p=0, attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_basis,
            attn_p=attn_p,
            proj_p=p
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )

    def forward(self, x):
        """Run forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape is `(batch_size, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape is `(batch_size, n_patches + 1, dim)`.

        """

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    