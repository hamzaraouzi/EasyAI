"""vit model."""

import torch.nn as nn
import torch
from .layers import EmbeddingStem, OutputLayer, Transformer
from ..abstractClassifier import AbstractClassifier


class VIT(AbstractClassifier):
    """Vit Module."""

    def __init__(
        self,
        image_size: tuple = (256, 256),
        path_size: tuple = (16, 16),
        in_channels: int = 3,
        embedding_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
    ) -> None:
        """vit class ini method.

        Args:
            image_size (tuple): _description_. Defaults to (224, 224).
            path_size (tuple): _description_. Defaults to (16, 16).
            in_channels (int): _description_. Defaults to 3.
            embedding_dim (int): _description_. Defaults to 768.
            num_layers (int): _description_. Defaults to 12.
            num_heads (int): _description_. Defaults to 12.
            qkv_bias (bool): _description_. Defaults to True.
            mlp_ratio (float): _description_. Defaults to 4.0.
            num_classes (int): _description_. Defaults to 10.
        """
        super(VIT, self).__init__()

        self.embedding_layer = EmbeddingStem(
            image_size=image_size,
            patch_size=path_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )

        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=0.0,
            qkv_bias=qkv_bias,
        )

        self.norm = nn.LayerNorm(embedding_dim)

        self.cls_layer = OutputLayer(
            embedding_dim=embedding_dim, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """vit forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.cls_layer(x)

    @staticmethod
    def prepareModel(model_name: str, num_classes: int, in_channels: int = 3):
        """VIT model preparation.

        Args:
            model_name (str): _description_
            num_classes (int): _description_
            in_channels (int): _description_. Defaults to 3.

        Returns:
            _type_: _description_
        """
        return VIT(in_channels=in_channels, num_classes=num_classes)
