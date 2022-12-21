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
