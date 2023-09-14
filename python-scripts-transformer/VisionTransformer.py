import torch
from torch import nn

from PatchEmbed import PatchEmbed
from Block import Block

class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision Transformer.

    Parameters
    ----------
    img_size : int
        Both height and the width of the image (ECG in our case). (This is 1D 5000)

    patch_size : int
        Both height and the width of the patch.

    in_chans : int
        Number of input channels. (This is 8)

    n_classes : int
        Number of classes to predict.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module relative to `embed_dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------

    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls_token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.

    """

    def __init__(
            self,
            img_size=5000,
            patch_size=50,
            in_chans=8,
            n_classes=1,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # zero tensors for the class token and the positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_basis=qkv_bias,
                p=p,
                attn_p=attn_p
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape is `(batch_size, in_chans, img_size)`.

        Returns
        -------
        logits: torch.Tensor
            Logits over all the classes - `(batch_size, n_classes)`.

        """

        n_samples = x.shape[0]
        x = self.patch_embed(x) # (batch_size, n_patches + 1, embed_dim)

        cls_token = self.cls_token.expand(n_samples, -1, -1) # (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (batch_size, n_patches + 1, embed_dim)
        x = x + self.pos_embed # (batch_size, n_patches + 1, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0] # just the cls token
        x = self.head(cls_token_final)

        x = torch.squeeze(x)

        return x
