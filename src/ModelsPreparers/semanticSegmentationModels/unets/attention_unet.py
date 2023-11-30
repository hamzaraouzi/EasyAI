"""Attention Unet model Implementation."""
import torch.nn as nn
import torch
from ..abstractSegmenter import AbstrctSegmenter

from abc import abstractmethod


class conv_block(nn.Module):
    """Convolution block class."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """constructor method for Convolution class.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
        """
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for conv_block Module.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """Up conv module."""

    def __init__(self, in_channels: int, out_channels: int):
        """constructor method  for up conv.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
        """
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """Attention block module."""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        """constructor of Attention_block class.

        Args:
            F_g (int): _description_
            F_l (int): _description_
            F_int (int): _description_
        """
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """forward method for Attention block module.

        Args:
            g (torch.Tensor): gating signal.
            x (torch.Tensor): encoder output.

        Returns:
            torch.Tensor: output tensor.
        """
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


class Attention_unet(AbstrctSegmenter):
    """Attention Unet module."""

    def __init__(self, num_classes: int, model_name: str, in_channels: int):
        """constructor method of Attention Unet module.

        Args:
            num_classes (int): number of classes.
            model_name (str): model name.
            in_channels (int): input channels.
        """
        super().__init__(
            num_classes=num_classes, model_name=model_name, in_channels=in_channels
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_channels=self.in_channels, out_channels=64)
        self.Conv2 = conv_block(in_channels=64, out_channels=128)
        self.Conv3 = conv_block(in_channels=128, out_channels=256)
        self.Conv4 = conv_block(in_channels=256, out_channels=512)
        self.Conv5 = conv_block(in_channels=512, out_channels=1024)

        self.Up5 = up_conv(in_channels=1024, out_channels=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(in_channels=1024, out_channels=512)

        self.Up4 = up_conv(in_channels=512, out_channels=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(in_channels=512, out_channels=256)

        self.Up3 = up_conv(in_channels=256, out_channels=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(in_channels=256, out_channels=128)

        self.Up2 = up_conv(in_channels=128, out_channels=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(in_channels=128, out_channels=64)

        self.Conv_1x1 = nn.Conv2d(
            64, self.num_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for Attention Unet.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x1 = self.Conv1(x)

        x2 = self.max_pool(x1)
        x2 = self.Conv2(x2)

        x3 = self.max_pool(x2)
        x3 = self.Conv3(x3)

        x4 = self.max_pool(x3)
        x4 = self.Conv4(x4)

        x5 = self.max_pool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        if self.num_classes == 1:
            d1 = d1.squeeze(1)
        return d1

    @staticmethod
    def prepareModel(
        model_name: str, in_channels: int = 3, num_classes: int = 10, **kwargs: dict
    ) -> nn.Module:
        """Desired model preparation.

        Args:
            model_name (str): model name.
            in_channels (int): input channels.
            num_classes (int): number of classes.
            kwargs (dict): _description_.

        Returns:
            nn.Module: _description_
        """
        return Attention_unet(
            model_name=model_name, in_channels=in_channels, num_classes=num_classes
        )
