"""This model contains implementation of different sizes of resnet."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstractClassifier import AbstractClassifier


class Residual_blockC(nn.Module):
    """residual block."""

    def __init__(
        self,
        in_channels: int,
        intermidiate_channels: int,
        expand=False,
        downsample=False,
    ):
        """Init method for Residual_blockC class.

        Args:
            in_channels (int): _description_
            intermidiate_channels (int): _description_
            expand (bool, optional): _description_. Defaults to False.
            downsample (bool, optional): _description_. Defaults to False.
        """
        super(Residual_blockC, self).__init__()
        self.in_channels = in_channels
        self.downsample = downsample
        self.intermidiate_channels = intermidiate_channels
        self.expand = expand

        stride = 2 if self.downsample else 1
        if self.expand:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    self.intermidiate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.intermidiate_channels * 4),
            )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.intermidiate_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.intermidiate_channels),
            nn.ReLU(),
            nn.Conv2d(
                self.intermidiate_channels,
                self.intermidiate_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.intermidiate_channels),
            nn.ReLU(),
            nn.Conv2d(
                self.intermidiate_channels,
                self.intermidiate_channels * 4,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.intermidiate_channels * 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass of Residual_blockC.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            ouput tensor
        """
        out = self.conv_layers(x)
        out_1 = self.projection(x) if self.expand else x.clone()

        return F.relu(out + out_1)


class Resnet101(AbstractClassifier):
    """Resnet101 class."""

    def __init__(self, model_name: str, in_channels: int = 3, num_classes: int = 10):
        """Init method for Resnet101 class.

        Args:
            model_name (str): model name.
            in_channels (int): input channels . Defaults to 3.
            num_classes (int): number of classes. Defaults to 10.
        """
        super(Resnet101, self).__init__(model_name=model_name, num_classes=num_classes)
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.b64 = nn.ModuleList([])
        self.b64.append(
            Residual_blockC(
                in_channels=64, intermidiate_channels=64, expand=True, downsample=False
            )
        )
        self.b64.append(
            Residual_blockC(
                in_channels=256,
                intermidiate_channels=64,
                expand=False,
                downsample=False,
            )
        )
        self.b64.append(
            Residual_blockC(
                in_channels=256,
                intermidiate_channels=64,
                expand=False,
                downsample=False,
            )
        )

        self.b128 = nn.ModuleList([])
        self.b128.append(
            Residual_blockC(
                in_channels=256, intermidiate_channels=128, expand=True, downsample=True
            )
        )
        for _ in range(3):
            self.b128.append(
                Residual_blockC(
                    in_channels=128 * 4,
                    intermidiate_channels=128,
                    expand=False,
                    downsample=False,
                )
            )

        self.b256 = nn.ModuleList([])
        self.b256.append(
            Residual_blockC(
                in_channels=512, intermidiate_channels=256, expand=True, downsample=True
            )
        )
        for _ in range(22):
            self.b256.append(
                Residual_blockC(
                    in_channels=256 * 4,
                    intermidiate_channels=256,
                    expand=False,
                    downsample=False,
                )
            )

        self.b512 = nn.ModuleList([])
        self.b512.append(
            Residual_blockC(
                in_channels=1024,
                intermidiate_channels=512,
                expand=True,
                downsample=True,
            )
        )
        self.b512.append(
            Residual_blockC(
                in_channels=512 * 4,
                intermidiate_channels=512,
                expand=False,
                downsample=False,
            )
        )
        self.b512.append(
            Residual_blockC(
                in_channels=512 * 4,
                intermidiate_channels=512,
                expand=False,
                downsample=False,
            )
        )

        self.fc = nn.Linear(2048 * 1 * 1, self.num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass for Resnet101.

        Args:
            x (torch.Tensor): bacth of examples

        Returns:
            torch.Tensor:  batched predictions
        """
        x = self.initial_block(x)
        for block in self.b64:
            x = block(x)

        for block in self.b128:
            x = block(x)

        for block in self.b256:
            x = block(x)

        for block in self.b256:
            x = block(x)

        for block in self.b512:
            x = block(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        self.fc(x)
        return x

    @staticmethod
    def prepareModel(
        model_name: str, in_channels: int = 3, num_classes: int = 10
    ) -> nn.Module:
        """Prepare Resnet101 model.

        Args:
            model_name (str): _description_
            in_channels (int): _description_. Defaults to 3.
            num_classes (int): _description_. Defaults to 10.

        Returns:
            nn.Module: _description_
        """
        return Resnet101(
            model_name=model_name, in_channels=in_channels, num_classes=num_classes
        )


class Residual_blockB(nn.Module):
    """Residual Block."""

    def __init__(self, in_channels: int, out_channels: int, downsample=False) -> None:
        """Initial Residual Block.

        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            downsample (bool, optional): downsample? . Defaults to False.
        """
        super(Residual_blockB, self).__init__()
        self.downsample = downsample
        stride = 1
        if self.downsample:
            stride = 2
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """forward pass of Residual_blockB.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            ouput tensor
        """
        out = self.conv_layers(x)
        out_2 = self.projection(x) if self.downsample else x.clone()
        return F.relu(out + out_2)


class Resnet34(AbstractClassifier):
    """Implementation Resnet34."""

    def __init__(self, model_name: str, in_channels: int = 3, num_classes: int = 10):
        """Init method for Resnet101 class.

        Args:
            model_name (str): model_name.
            in_channels (int): input channels . Defaults to 3.
            num_classes (int): number of classes. Defaults to 10.
        """
        super(Resnet34, self).__init__(model_name=model_name, num_classes=num_classes)

        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc = nn.Linear(512, self.num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.b64 = nn.ModuleList([])
        self.b64.append(
            Residual_blockB(in_channels=64, out_channels=64, downsample=False)
        )
        self.b64.append(
            Residual_blockB(in_channels=64, out_channels=64, downsample=False)
        )
        self.b64.append(
            Residual_blockB(in_channels=64, out_channels=64, downsample=False)
        )

        self.b128 = nn.ModuleList([])
        self.b128.append(
            Residual_blockB(in_channels=64, out_channels=128, downsample=True)
        )
        self.b128.append(
            Residual_blockB(in_channels=128, out_channels=128, downsample=False)
        )
        self.b128.append(
            Residual_blockB(in_channels=128, out_channels=128, downsample=False)
        )
        self.b128.append(
            Residual_blockB(in_channels=128, out_channels=128, downsample=False)
        )

        self.b256 = nn.ModuleList([])
        self.b256.append(
            Residual_blockB(in_channels=128, out_channels=256, downsample=True)
        )
        for _ in range(5):
            self.b256.append(
                Residual_blockB(in_channels=256, out_channels=256, downsample=False)
            )

        self.b512 = nn.ModuleList([])
        self.b512.append(
            Residual_blockB(in_channels=256, out_channels=512, downsample=True)
        )
        for _ in range(2):
            self.b512.append(
                Residual_blockB(in_channels=512, out_channels=512, downsample=False)
            )

    def forward(self, x):
        """forward method resenet34 Module.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            ouput tensor
        """
        x = self.initial_block(x)

        for block in self.b64:
            x = block(x)

        for block in self.b128:
            x = block(x)

        for block in self.b256:
            x = block(x)

        for block in self.b512:
            x = block(x)

        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    @staticmethod
    def prepareModel(model_name: str, num_classes: int) -> nn.Module:
        """prepare resnet34 model.

        Args:
            model_name (str): _description_
            num_classes (int): _description_

        Returns:
            nn.Module: _description_
        """
        return Resnet34(model_name=model_name, in_channels=3, num_classes=num_classes)
