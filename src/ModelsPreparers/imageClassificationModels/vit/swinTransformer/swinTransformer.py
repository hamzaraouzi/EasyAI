"""Swin Transformer Model."""
from .layers import StageModule
from ...abstractClassifier import AbstractClassifier
from torch import nn
import torch


class SwinTransformer(AbstractClassifier):
    """SwinTransfoermer Model."""

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        layers: tuple,
        heads: tuple,
        in_channels: int = 3,
        num_classes: int = 10,
        head_dim: int = 32,
        window_size: int = 7,
        downscaling_factors: tuple = (4, 2, 2, 2),
    ) -> None:
        """init method.

        Args:
            model_name (str): _description_
            hidden_dim (int): _description_
            layers (tuple): _description_
            heads (tuple): _description_
            in_channels (int): _description_. Defaults to 3.
            num_classes (int): _description_. Defaults to 10.
            head_dim (int): _description_. Defaults to 32.
            window_size (int): _description_. Defaults to 7.
            downscaling_factors (tuple): _description_. Defaults to (4, 2, 2, 2).
        """
        super().__init__(model_name, num_classes)
        self.model_name = model_name
        self.num_classes = num_classes

        self.stage1 = StageModule(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            head_dim=head_dim,
            num_heads=heads[0],
            window_size=window_size,
        )

        self.stage2 = StageModule(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim * 2,
            layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            head_dim=head_dim,
            num_heads=heads[1],
            window_size=window_size,
        )

        self.stage3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dim=hidden_dim * 4,
            layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            head_dim=head_dim,
            num_heads=heads[2],
            window_size=window_size,
        )

        self.stage4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dim=hidden_dim * 8,
            layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            head_dim=head_dim,
            num_heads=heads[3],
            window_size=window_size,
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8), nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        x = self.mlp(x)
        return x

    @staticmethod
    def prepareModel(model_name: str, num_classes: int = 10) -> nn.Module:
        """prepare model.

        Args:
            model_name (str): _description_
            num_classes (int): _description_. Defaults to 10.

        Returns:
            nn.Module: _description_
        """
        if model_name == "swin-t":
            return SwinTransformer(
                hidden_dim=96,
                layers=(2, 2, 6, 2),
                heads=(3, 6, 12, 24),
                model_name=model_name,
            )

        elif model_name == "swin-s":
            return SwinTransformer(
                hidden_dim=96,
                layers=(2, 2, 18, 2),
                heads=(3, 6, 12, 24),
                model_name=model_name,
            )

        elif model_name == "swin-b":
            return SwinTransformer(
                hidden_dim=128,
                layers=(2, 2, 18, 2),
                heads=(4, 8, 16, 32),
                model_name=model_name,
            )

        elif model_name == "swin-l":
            return SwinTransformer(
                hidden_dim=192,
                layers=(2, 2, 18, 2),
                heads=(6, 12, 24, 48),
                model_name=model_name,
            )
