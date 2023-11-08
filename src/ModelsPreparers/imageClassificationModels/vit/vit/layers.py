"""Vit components."""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class EmbeddingStem(nn.Module):
    """Embeddings."""

    def __init__(
        self,
        image_size: tuple = (224, 224),
        patch_size: tuple = (16, 16),
        in_channels: int = 3,
        embedding_dim: int = 768,
    ) -> None:
        """EmbeddingStem init method.

        Args:
            image_size (tuple): _description_. Defaults to (224, 224).
            patch_size (tuple): _description_. Defaults to (16, 16).
            in_channels (int): _description_. Defaults to 3.
            embedding_dim (int): _description_. Defaults to 768.
        """
        super(EmbeddingStem, self).__init__()

        image_height, image_width = image_size[0], image_size[1]
        patch_height, patch_width = patch_size[0], patch_size[1]
        self.grid_size = (
            image_height // patch_height,
            image_width // patch_width,
        )
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        num_patches += 1

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embedding_dim))
        patch_dim = in_channels * patch_height * patch_width
        self.projection = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """EmbeddingStem forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.projection(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        return x + self.pos_embed


class PreNorm(nn.Module):
    """Normalization."""

    def __init__(self, dim: int, fn: nn.Module) -> None:
        """PreNorm init method.

        Args:
            dim (int): _description_
            fn (nn.Module): _description_
        """
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs):
        """prenorm forward method.

        Args:
            x (torch.Tensor): _description_
            **kwargs: other params.

        Returns:
            _type_: _description_
        """
        return self.fn(self.norm(x), **kwargs)


class Attenion(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Attention init method.

        Args:
            dim (int): _description_
            num_heads (int): _description_. Defaults to 8.
            qkv_bias (bool): _description_. Defaults to False.
            attn_drop (float): _description_. Defaults to 0.0.
            proj_drop (float): _description_. Defaults to 0.0.
        """
        super(Attenion, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 1 / head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout2d(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout2d(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """Feed forward module."""

    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float = 0.0) -> None:
        """Feed forward init.

        Args:
            dim (int): _description_
            hidden_dim (int): _description_
            dropout_rate (float): _description_. Defaults to 0.0.
        """
        super(FeedForward, self).__init__()
        # Original: https://arxiv.org/pdf/2010.11929.pdf
        l = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
        ]
        self.layers = nn.Sequential(*l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """OutputLayer forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.layers(x)


class OutputLayer(nn.Module):
    """OutputLayer class."""

    def __init__(self, embedding_dim: int, num_classes: int = 10) -> None:
        """OutputLayer init method.

        Args:
            embedding_dim (int): _description_
            num_classes (int): _description_. Defaults to 10.
        """
        super(OutputLayer, self).__init__()
        self.layer = nn.Linear(embedding_dim, num_classes)
        self.to_cls_token = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """OutputLayer forward method.

        Args:
            x (torch.Tensor): _description_.

        Returns:
            torch.Tensor: _description_.
        """
        x = self.to_cls_token(x[:, 0])
        return self.layer(x)


class Transformer(nn.Module):
    """transformer class."""

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout=0.0,
        qkv_bias=True,
    ) -> None:
        """init method for Transformer..

        Args:
            dim (int): _description_
            depth (int): _description_
            heads (int): _description_
            mlp_ratio (float): _description_. Defaults to 4.0 .
            attn_dropout (float): _description_. Defaults to 0.0.
            dropout (float): _description_. Defaults to 0.0.
            qkv_bias (bool): _description_. Defaults to True.
        """
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])

        mlp_dim = int(mlp_ratio * dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attenion(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate=dropout)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        for (
            attn,
            ff,
        ) in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x
