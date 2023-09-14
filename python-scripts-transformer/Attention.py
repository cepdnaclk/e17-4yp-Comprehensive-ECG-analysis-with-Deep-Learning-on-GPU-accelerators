from torch import nn

class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        Last dimension of the input tensors (embed_dim).
        The input and out dimension of per token features

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------

    scale : float
        Normalizing constant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.

    """


    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        # define the dimentionality of each of the heads
        self.head_dim = dim // n_heads
        # when we concatonate all the heads we should get the same dim as the input
        
        # from attention is all you need paper
        self.scale = dim ** -0.5

        # get an embedding and output q, k, v
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)

        # get the concatenated output of all the heads and maps to a new mapping
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

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

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        # n_patches + 1 --> class token as the first token
        qkv = self.qkv(x)   # (batch_size, n_patches + 1, 3 * dim) x is mulit dimensional
        qkv = qkv.reshape(n_samples,n_tokens,3,self.n_heads,self.head_dim) # (batch_size, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2,0,3,1,4)    # (3, batch_size, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2,-1)    # (batch_size, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1) # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v # (batch_size, n_heads, n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1,2) # (batch_size, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (batch_size, n_patches + 1, dim)

        x = self.proj(weighted_avg) # (batch_size, n_patches + 1, dim)
        x = self.proj_drop(x) # (batch_size, n_patches + 1, dim)
        #print(x.shape)
        return x
