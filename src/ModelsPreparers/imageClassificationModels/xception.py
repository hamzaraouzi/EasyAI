"""Xception architecture implementation."""
import torch
import torch.nn as nn


class Separable(nn.Module):
    """Separable Convolution."""

    def __init__(self, in_channels: int, out_1x1: int, out_channels: int) -> None:
        """init method for Separable model.

        Args:
            in_channels (int): _description_
            out_1x1 (int): _description_
            out_channels (int): _description_
        """
        super(Separable, self).__init__()
        self.pointwise_conv = nn.Conv2d(in_channels, out_1x1, kernel_size=1, stride=1)
        self.deptwise_conv = nn.Conv2d(
            out_1x1, out_channels, kernel_size=3, stride=1, padding=1, groups=out_1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.pointwise_conv(x)
        x = self.deptwise_conv(x)
        return x


class ConvBlock(nn.Module):
    """ConvBlock class."""

    def __init__(self, in_channels: int, out_channels: int, **kwrags) -> None:
        """Init method of ConvBklock class.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            **kwrags : args for nn.Convolution.
        """
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwrags)
        self.bn = nn.BatchNorm2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method of ConvBlock.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return self.relu(self.bn(self.conv(x)))


class EntryFlow(nn.Module):
    """EntryFlow block."""

    def __init__(self, in_channels: int = 3) -> None:
        """EntryFlow init method.

        Args:
            in_channels (int): input channels. Defaults to 3.
        """
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=3, stride=2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128)
        )

        self.block1 = nn.Sequential(
            Separable(64, 32, 128),
            nn.ReLU(),
            Separable(128, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2), nn.BatchNorm2d(256)
        )

        self.block2 = nn.Sequential(
            nn.ReLU(),
            Separable(128, 64, 256),
            nn.ReLU(),
            Separable(256, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.res3 = nn.Sequential(
            nn.Conv2d(256, 728, kernel_size=1, stride=2), nn.BatchNorm2d(728)
        )

        self.block3 = nn.Sequential(
            nn.ReLU(),
            Separable(256, 182, 728),
            nn.ReLU(),
            Separable(728, 364, 728),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method of EntryFlow.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.conv1(x)
        x_res = self.res1(x)
        x_block = self.block1(x)
        x = x_res.add(x_block)

        x_res = self.res2(x)
        x_block = self.block2(x)
        x = x_block.add(x_res)

        x_res = self.res3(x)
        x_block = self.block3(x)
        x = x_block.add(x_res)
        return x


class MiddleFlow(nn.Module):
    """MiddleFlow class."""

    def __init__(self, in_channels: int = 728):
        """the init method of the Middleflow module.

        Args:
            in_channels (int): input channels. Defaults to 728.
        """
        super(MiddleFlow, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            Separable(in_channels, 364, in_channels),
            nn.ReLU(),
            Separable(in_channels, 364, in_channels),
            nn.ReLU(),
            Separable(in_channels, 364, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward methdod of middleflow module.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        return x.add(self.layers(x))
