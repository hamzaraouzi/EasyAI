"""pspnet model implementation."""
import torch
from torch import nn
from torch.nn import functional as F
from ..abstractSegmenter import AbstrctSegmenter
from ...imageClassificationModels.abstractClassifier import AbstractClassifier
import ModelsPreparers.semanticSegmentationModels.decodersSegmentors.extractors as extractors
import extractors


class PSPModule(nn.Module):
    """PSPModule class."""

    def __init__(self, features: int, out_features: int = 1024):
        """init method.

        Args:
            features (int): _description_
            out_features (int): _description_. Defaults to 1024.
        """
        super().__init__()
        self.stage_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(features, features, kernel_size=1, bias=False),
        )
        self.stage_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Conv2d(features, features, kernel_size=1, bias=False),
        )
        self.stage_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Conv2d(features, features, kernel_size=1, bias=False),
        )
        self.stage_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Conv2d(features, features, kernel_size=1, bias=False),
        )
        self.bottleneck = nn.Conv2d(
            features * 5, out_features, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        _, _, h, w = x.shape
        s1 = F.upsample(self.stage_1(x), size=(h, w), mode="bilinear")
        s2 = F.upsample(self.stage_2(x), size=(h, w), mode="bilinear")
        s3 = F.upsample(self.stage_3(x), size=(h, w), mode="bilinear")
        s4 = F.upsample(self.stage_4(x), size=(h, w), mode="bilinear")
        out = self.bottleneck(torch.cat([x, s1, s2, s3, s4], dim=1))
        return F.relu(out)


class Upsample(nn.Module):
    """PSPUpsample Module."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """init method.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PSPUpsample Module.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        _, _, h, w = x.shape
        x = F.upsample(x, size=(h * 2, w * 2), mode="bilinear")
        return self.conv(x)


class PSPNet(AbstrctSegmenter):
    """PSPNet Module."""

    def __init__(
        self,
        model_name: str,
        backbone: AbstractClassifier,
        psp_size: int,
        feats_id: int,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        """init method.

        Args:
            model_name (str): _description_
            backbone (AbstractClassifier): _description_
            psp_size (int): _description_
            feats_id (int): _description_
            in_channels (int): _description_. Defaults to 3.
            num_classes (int): _description_. Defaults to 10.
        """
        super().__init__(in_channels, num_classes, model_name)
        self.psp_size = psp_size
        self.backbone = backbone
        self.feats_id = feats_id
        self.psp = PSPModule(features=psp_size, out_features=1024)

        self.up_1 = Upsample(1024, 256)
        self.up_2 = Upsample(256, 64)
        self.up_3 = Upsample(64, 64)
        self.up_4 = Upsample(64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1), nn.LogSoftmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method.

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        feats = self.backbone.extract_features(x)
        f = feats[self.feats_id]

        assert (
            f.shape[1] == self.psp_size
        ), f"inconsistant psp_size={self.psp_size} with features_dim={f.shape[1]}"
        p = self.psp(f)

        p = self.up_1(p)
        p = self.up_2(p)
        p = self.up_3(p)
        p = self.up_4(p)

        p = self.final(p)
        p = F.upsample(p, size=x.shape[2:], mode="bilinear")
        return p

    @staticmethod
    def prepareModel(
        model_name: str, in_channels: int = 3, num_classes: int = 10, **kwargs: dict
    ) -> nn.Module:
        """prepare pspnet.

        Args:
            model_name (str): _description_
            in_channels (int): _description_. Defaults to 3.
            num_classes (int): _description_. Defaults to 10.
            kwargs (dict): _description_.

        Returns:
            nn.Module: _description_
        """
        key_map = {
            "xception": {"feats_id": 2, "psp_size": 2048},
            "resnet34": {"feats_id": 3, "psp_size": 256},
            "resnet101": {"feats_id": 3, "psp_size": 1024},
            "mobileNetV1": {"feats_id": 13, "psp_size": 96},
            "mobileNetV2": {"feats_id": 13, "psp_size": 96},
            "mobileNetV3": {"feats_id": 9, "psp_size": 48},
        }
        backbone = getattr(extractors, kwargs["backbone"])(
            model_name=kwargs["backbone"], num_classes=num_classes
        )
        params = key_map.get(kwargs["backbone"], None)

        assert params is not None, f"{backbone} not available for pspnet as backbone"

        feats_id, psp_size = params["feats_id"], params["psp_size"]
        return PSPNet(
            model_name=model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            backbone=backbone,
            feats_id=feats_id,
            psp_size=psp_size,
        )
