"""Classification Model is a class that will manage loading and preparation of desired model."""
import torch.nn as nn
import yaml

from .imageClassificationModels.cnn.mobileNetv1 import MobileNetV1
from .imageClassificationModels.cnn.mobileNetv2 import MobileNetV2
from .imageClassificationModels.cnn.mobileNetv3 import MobileNetV3
from .imageClassificationModels.cnn.resnet import Resnet101, Resnet34
from .imageClassificationModels.cnn.xception import Xception
from .imageClassificationModels.vit.vit.vit import VIT
from .imageClassificationModels.vit.swinTransformer.swinTransformer import (
    SwinTransformer,
)


from .semanticSegmentationModels.unets.unet import UNET
from .semanticSegmentationModels.unets.attention_unet import Attention_unet
from .semanticSegmentationModels.unets.r2_unet import R2U_net
from .semanticSegmentationModels.unets.r2_attention_unet import R2AttU_net


class ModelFactory:
    """Classification Model is a class that will manage loading and preparation of desired modelll."""

    def __init__(self, config_path: str) -> None:
        """Init method for classificationModel class.

        Args:
            config_path (str): the path to the yaml config path
        """
        super(ModelFactory, self).__init__()
        params2values = self.load_check_conf_file(config_path)

        self.model_name = params2values["name"]
        self.task = params2values["task"]
        self.pretrained = params2values.get("pretrained", False)
        self.num_classes = params2values["num_classes"]
        self.backbone = params2values.get("backbone", None)

    def load_check_conf_file(self, config_path: str) -> dict:
        """Loading desired model configuration from  yaml file.

        Args:
            config_path (str): the path to the yaml config path.

        Returns:
            dict: _description_
        """
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)

        params2values = {}
        for d in conf_values["model"]:
            for key, values in zip(d.keys(), d.values()):
                params2values[key] = values

        return params2values

    def prepareModels(self) -> nn.Module:
        """prepare  model.

        Raises:
            NotImplementedError: _description_.

        Returns:
            nn.Module: pytorch model.
        """
        if self.model_name == "vit":
            return VIT.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "mobileNetV1":
            return MobileNetV1.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "mobileNetV2":
            return MobileNetV2.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name in ["mobileNetV3-large", "mobileNetV3-small"]:
            return MobileNetV3.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "resnet101":
            return Resnet101.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "resnet34":
            return Resnet34.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "xception":
            return Xception.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "u-net":
            return UNET.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "attention-unet":
            return Attention_unet.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "r2-unet":
            return R2U_net.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "r2-attention-unet":
            return R2AttU_net.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name in ["swin-t", "swin-s", "swin-b", "swin-l"]:
            return SwinTransformer.prepareModel(
                model_name=self.model_name, num_classes=self.num_classes
            )

        if self.model_name == "pspnet":

            return PSPNet.prepareModel(
                model_name=self.model_name,
                num_classes=self.num_classes,
                backbone=self.backbone,
            )

        else:
            raise NotImplementedError(
                f"{self.model_name} not implemented, please don't hesitate to create an issue on our repositry, to think about adding this model"
            )

    def __call__(self) -> nn.Module:
        """prepare classification model.

        Returns:
            AbstractClassifier: classification model.
        """
        return self.prepareModels()
