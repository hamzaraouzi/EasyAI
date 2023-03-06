"""Unet Implementation."""
from abc import abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from .abstractSegmenter import AbstrctSegmenter


class UNET(AbstrctSegmenter):
    """Unet module class."""

    def __init__(self, model_name: str, in_channels: int = 3, num_classes: int = 1):
        """init method for unet model.

        Args:
            model_name (str): model name.
            in_channels (int): input channels. Defaults to 3.
            num_classes (int): number of classes. Defaults to 1.
        """
        super(UNET, self).__init__(
            model_name=model_name, num_classes=num_classes, in_channels=in_channels
        )
        self.layers = [in_channels, 64, 128, 256, 512, 1024]

        self.double_conv_downs = nn.ModuleList(
            [
                self.__double_conv(layer, layer_n)
                for layer, layer_n in zip(self.layers[:-1], self.layers[1:])
            ]
        )

        self.up_trans = nn.ModuleList(
            [
                nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2)
                for layer, layer_n in zip(
                    self.layers[::-1][:-2], self.layers[::-1][1:-1]
                )
            ]
        )

        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]]
        )

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def __double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """generation convolution block for unet.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.

        Returns:
            nn.Sequentail: convolution block.
        """
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for unet.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor:  output tensor.
        """
        # down layers
        concat_layers = []
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool_2x2(x)

        concat_layers = concat_layers[::-1]

        # up layers
        for up_trans, double_conv_up, concat_layer in zip(
            self.up_trans, self.double_conv_ups, concat_layers
        ):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:])

            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)

        x = self.final_conv(x)

        if self.num_classes == 1:
            x = x.squeeze(1)
        return x

    @abstractmethod
    def prepareModel(
        model_name: str, in_channels: int = 3, num_classes: int = 10
    ) -> nn.Module:
        """Desired model preparation.

        Args:
            in_channels (int): input channels.
            num_classes (int): number of classes.

        Returns:
            nn.Module: _description_
        """
        return UNET(
            model_name=model_name, in_channels=in_channels, num_classes=num_classes
        )
