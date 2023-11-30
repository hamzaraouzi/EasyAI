"""Xception architecture implementation."""
from typing import List
import torch
import torch.nn as nn
from ..abstractClassifier import AbstractClassifier


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
        self.bn = nn.BatchNorm2d(out_channels)

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


class ExitFlow(nn.Module):
    """ExitFlow Module."""

    def __init__(self, in_channels: int, num_classes: int = 10):
        """init method of ExitFlow.

        Args:
            in_channels (int): input channels.
            num_classes (int): number of classes. Defaults to 10.
        """
        super(ExitFlow, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(728, 1024, kernel_size=1, stride=2), nn.BatchNorm2d(1024)
        )

        self.block = nn.Sequential(
            nn.ReLU(),
            Separable(728, 364, 728),
            nn.ReLU(),
            Separable(728, 512, 1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.tail = nn.Sequential(
            Separable(1024, 768, 1536),
            nn.ReLU(),
            Separable(1536, 1024, 2048),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method of Exitflow.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x_res = self.res(x)
        x = x_res.add(self.block(x))
        x = self.tail(x)
        return x


class Xception(AbstractClassifier):
    """Xception Module."""

    def mFlows(self) -> nn.Sequential:
        """method for creation a sequence of MiddleFLow blocks.

        Returns:
            nn.Sequential: sequence of MiddleFLow blocks
        """
        layers = []
        for i in range(8):
            layers.append(MiddleFlow())

        return nn.Sequential(*layers)

    def __init__(self, model_name: str, in_channels: int = 3, num_classes: int = 10):
        """init method of Xception module.

        Args:
            model_name (str): model name.
            in_channels (int): input channels. Defaults to 3.
            num_classes (int): num classes. Defaults to 10.
        """
        super(Xception, self).__init__(model_name=model_name, num_classes=num_classes)

        self.entry = EntryFlow(in_channels=in_channels)

        self.middle_flow = self.mFlows()

        self.exit_flow = ExitFlow(728, num_classes)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """features extraction method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            List[torch.Tensor]: _description_
        """
        feats = []
        x = self.entry(x)
        feats.append(x)
        x = self.middle_flow(x)
        feats.append(x)
        x = self.exit_flow(x)
        feats.append(x)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor.
        """
        x = self.entry(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def prepareModel(
        model_name: str, num_classes: int, in_channels: int = 3, **kwargs: dict
    ) -> nn.Module:
        """MobileNet model preparation.

        Args:
            model_name (str): _description_
            num_classes (int): _description_
            in_channels (int): _description_. Defaults to 3.
            kwargs (dict): _description_.

        Returns:
            nn.Module: _description_
        """
        return Xception(
            model_name=model_name, in_channels=in_channels, num_classes=num_classes
        )
