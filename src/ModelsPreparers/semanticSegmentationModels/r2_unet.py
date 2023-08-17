"""implementation of r2-unet and r2-attention-unet."""
import torch
import torch.nn as nn
from .abstractSegmenter import AbstrctSegmenter


class Recurrent_block(nn.Module):
    """Recurrent convolution block."""

    def __init__(self, channels: int, t: int = 2) -> None:
        """constructor method for Recurrent_block class.

        Args:
            channels (int): in/out channels.
            t (int): recurration times.
        """
        super(Recurrent_block, self).__init__()
        self.t = t
        self.channels = channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=True
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for Recurrent convolution block.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    """RRCNN block."""

    def __init__(self, in_channels: int, out_channels: int, t=2) -> None:
        """RCNN block constructor metho.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            t (int, optional): recurration time. Defaults to 2.
        """
        super(RRCNN_block, self).__init__()
        self.rrcnn = nn.Sequential(
            Recurrent_block(channels=out_channels, t=t),
            Recurrent_block(channels=out_channels, t=t),
        )
        self.conv_1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for RRCNN block.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.conv_1x1(x)
        x1 = self.rrcnn(x)
        return x1 + x


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


class R2U_net(AbstrctSegmenter):
    """R2U_net class.."""

    def __init__(self, in_channels: int, num_classes: int, model_name: str):
        """_summary_.

        Args:
            in_channels (int): _description_
            num_classes (int): _description_
            model_name (str): _description_
        """
        super().__init__(in_channels, num_classes, model_name)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sample = nn.Upsample(scale_factor=2)

        self.rrcnn1 = RRCNN_block(in_channels=self.in_channels, out_channels=64)
        self.rrcnn2 = RRCNN_block(in_channels=64, out_channels=128)
        self.rrcnn3 = RRCNN_block(in_channels=128, out_channels=256)
        self.rrcnn4 = RRCNN_block(in_channels=256, out_channels=512)
        self.rrcnn5 = RRCNN_block(in_channels=512, out_channels=1024)

        self.up5 = up_conv(in_channels=1024, out_channels=512)
        self.up_rrcnn5 = RRCNN_block(in_channels=1024, out_channels=512)

        self.up4 = up_conv(in_channels=512, out_channels=256)
        self.up_rrcnn4 = RRCNN_block(in_channels=512, out_channels=256)

        self.up3 = up_conv(in_channels=512, out_channels=256)
        self.up_rrcnn3 = RRCNN_block(in_channels=256, out_channels=128)

        self.up2 = up_conv(in_channels=128, out_channels=64)
        self.up_rrcnn2 = RRCNN_block(in_channels=128, out_channels=64)

        self.conv_1x1 = nn.Conv2d(64, num_classes, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        x1 = self.rrcnn1(x)

        x2 = self.max_pool(x1)
        x2 = self.rrcnn2(x2)

        x3 = self.max_pool(x2)
        x3 = self.rrcnn3(x3)

        x4 = self.max_pool(x3)
        x4 = self.rrcnn4(x4)

        x5 = self.max_pool(x4)
        x5 = self.rrcnn5(x5)

        # decoding
        d5 = self.up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_rrcnn5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_rrcnn4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_rrcnn3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_rrcnn2(d2)

        d1 = self.conv_1x1(d2)
        return d1

    @staticmethod
    def prepareModel(
        model_name: str, in_channels: int = 3, num_classes: int = 10
    ) -> nn.Module:
        """Desired model preparation.

        Args:
            model_name (str): model_name.
            in_channels (int): input channels.
            num_classes (int): number of classes.

        Returns:
            nn.Module: _description_
        """
        return R2U_net(
            model_name=model_name, in_channels=in_channels, num_classes=num_classes
        )
