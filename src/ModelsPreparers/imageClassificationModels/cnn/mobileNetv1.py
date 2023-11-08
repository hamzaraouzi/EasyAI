"""MobileNetV1 pytorch Implementattion."""

import torch
import torch.nn as nn
from ..abstractClassifier import AbstractClassifier


class SeparableConv(nn.Module):
    """Separable Convolution implementation."""

    def __init__(self, in_channels: int, out_channels: int, stride=1) -> None:
        """Init method for Separable Convolution Module.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            stride (int, optional): _description_. Defaults to 1.
        """
        super(SeparableConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
                groups=in_channels,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward methiode for SeparableConv Module.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.layers(x)


class MobileNetV1(AbstractClassifier):
    """MobileNetV1 Class."""

    def __init__(
        self,
        model_name: str,
        in_channels: int = 3,
        shallow: bool = False,
        num_classes: int = 10,
    ) -> None:
        """Init method for MobileNetV1 Module.

        Args:
            model_name (str): model name.w
            in_channels (int): _description_. Defaults to 3.
            shallow (bool): _description_. Defaults to False.
            num_classes (int): _description_. Defaults to 10.
        """
        super(MobileNetV1, self).__init__(
            model_name=model_name, num_classes=num_classes
        )
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.ModuleList([])
        self.layers.append(SeparableConv(in_channels=32, out_channels=64, stride=1))
        self.layers.append(SeparableConv(in_channels=64, out_channels=128, stride=2))
        self.layers.append(SeparableConv(in_channels=128, out_channels=128, stride=1))
        self.layers.append(SeparableConv(in_channels=128, out_channels=256, stride=2))
        self.layers.append(SeparableConv(in_channels=256, out_channels=256, stride=1))
        self.layers.append(SeparableConv(in_channels=256, out_channels=512, stride=2))

        if not shallow:
            for _ in range(5):
                self.layers.append(
                    SeparableConv(in_channels=512, out_channels=512, stride=1)
                )

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for MobileNetV1.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x = self.initial_block(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    @staticmethod
    def prepareModel(model_name: str, num_classes: int = 10) -> nn.Module:
        """MobileNetV1 Model preparation.

        Args:
            model_name (str): _description_
            num_classes (int): _description_. Defaults to 10.

        Returns:
            nn.Module: _description_.
        """
        return MobileNetV1(
            model_name=model_name, in_channels=3, num_classes=num_classes
        )
