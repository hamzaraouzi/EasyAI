"""mobileNeV2 implemetation."""
import torch
import torch.nn as nn
import numpy as np
from ..abstractClassifier import AbstractClassifier


def conv_bn(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    """Convolution block.

    Args:
        in_channels (int): _description_
        out_channels (int): _description_
        stride (int): _description_

    Returns:
        nn.Module: _description_
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(in_channels: int, out_channels: int) -> nn.Module:
    """1x1 convolution Block.

    Args:
        in_channels (int): _description_
        out_channels (int): _description_

    Returns:
        nn.Module: _description_
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def make_divisible(x: int, divisible_by: int = 8) -> int:
    """_summary_.

    Args:
        x (int): _description_
        divisible_by (int): _description_. Defaults to 8.

    Returns:
        int: _description_
    """
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    """Inverted Residual block."""

    def __init__(
        self, in_channels: int, out_channels: int, expansion_ratio: int, stride: int
    ) -> None:
        """init method InvertedResidual module.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            expansion_ratio (int): _description_
            stride (int): _description_
        """
        super(InvertedResidual, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expansion_ratio)

        if expansion_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            )

        else:
            self.conv = nn.Sequential(
                # pointwise convolution
                nn.Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise convolution
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise linear convolution
                nn.Conv2d(
                    hidden_dim,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for InvertedResidual block.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.use_residual:
            return x + self.conv(x)

        else:
            return self.conv(x)


class MobileNetV2(AbstractClassifier):
    """MobileNetV2 Implemetation."""

    def __init__(
        self, model_name: str, in_channels=3, num_classes=10, width_multiplier=1.0
    ):
        """Init method of MobileNetV2 class.

        Args:
            model_name (str): model name.
            in_channels (int, optional): number of channels. Defaults to 3.
            num_classes (int, optional): numer of classes. Defaults to 10.
            width_multiplier (float, optional): width multiplier. Defaults to 1.0.
        """
        super(MobileNetV2, self).__init__(
            model_name=model_name, num_classes=num_classes
        )
        inverted_residual_parameters = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.last_channels_dim = (
            make_divisible(1280 * width_multiplier) if width_multiplier > 1.0 else 1280
        )

        self.features = [conv_bn(in_channels, 32, stride=2)]
        in_channels = 32

        for t, c, n, s in inverted_residual_parameters:
            out_channels = (
                make_divisible(c * width_multiplier) if width_multiplier > 1.0 else c
            )

            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(
                            in_channels, out_channels, stride=s, expansion_ratio=t
                        )
                    )
                else:
                    self.features.append(
                        InvertedResidual(
                            in_channels, out_channels, stride=1, expansion_ratio=t
                        )
                    )

                in_channels = out_channels

        self.features.append(conv_1x1_bn(in_channels, self.last_channels_dim))
        self.features = nn.Sequential(*self.features)
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(self.last_channels_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for MobileNet model.

        Args:
            x (torch.Tensor): input batch

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x

    @staticmethod
    def prepareModel(model_name: str, num_classes: int) -> nn.Module:
        """MobileNet model preparation.

        Args:
            model_name (str): Model name.
            num_classes (int): numer of classes.

        Returns:
            nn.Model: _description_
        """
        return MobileNetV2(model_name=model_name, num_classes=num_classes)
