"""Attention R2U-net implementation."""
import torch
from torch import nn
from ..abstractSegmenter import AbstrctSegmenter

from .r2_unet import up_conv, Recurrent_block, RRCNN_block
from .attention_unet import Attention_block

from abc import abstractmethod


class R2AttU_net(AbstrctSegmenter):
    """R2AttU_net class."""

    def __init__(self, in_channels: int, num_classes: int, model_name: str):
        """_summary_.

        Args:
            in_channels (int): _description_
            num_classes (int): _description_
            model_name (str): _description_
        """
        super(R2AttU_net, self).__init__(
            in_channels=in_channels, num_classes=num_classes, model_name=model_name
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sample = nn.Upsample(scale_factor=2)

        self.rrcnn1 = RRCNN_block(in_channels=self.in_channels, out_channels=64)
        self.rrcnn2 = RRCNN_block(in_channels=64, out_channels=128)
        self.rrcnn3 = RRCNN_block(in_channels=128, out_channels=256)
        self.rrcnn4 = RRCNN_block(in_channels=256, out_channels=512)
        self.rrcnn5 = RRCNN_block(in_channels=512, out_channels=1024)

        self.up5 = up_conv(in_channels=1024, out_channels=512)
        self.att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.up_rrcnn5 = RRCNN_block(in_channels=1024, out_channels=512)

        self.up4 = up_conv(in_channels=512, out_channels=256)
        self.att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.up_rrcnn4 = RRCNN_block(in_channels=512, out_channels=256)

        self.up3 = up_conv(in_channels=256, out_channels=128)
        self.att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.up_rrcnn3 = RRCNN_block(in_channels=256, out_channels=128)

        self.up2 = up_conv(in_channels=128, out_channels=64)
        self.att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.up_rrcnn2 = RRCNN_block(in_channels=128, out_channels=64)

        self.conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

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

        # decoding.
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_rrcnn5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_rrcnn4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_rrcnn3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_rrcnn2(d2)

        d2 = self.conv_1x1(d2)

        if self.num_classes == 1:
            d2 = d2.squeeze(1)
        return d2

    @staticmethod
    def prepareModel(
        model_name: str, in_channels: int = 3, num_classes: int = 10, **kwargs: dict
    ) -> nn.Module:
        """Desired model preparation.

        Args:
            model_name (str): model_name.
            in_channels (int): input channels.
            num_classes (int): number of classes.
            kwargs (dict): _description_

        Returns:
            nn.Module: _description_
        """
        return R2AttU_net(
            model_name=model_name, in_channels=in_channels, num_classes=num_classes
        )
