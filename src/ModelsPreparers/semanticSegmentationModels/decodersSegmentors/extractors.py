"""backbone for two stages segmentation."""
from ...imageClassificationModels.cnn.mobileNetv1 import MobileNetV1
from ...imageClassificationModels.cnn.mobileNetv2 import MobileNetV2
from ...imageClassificationModels.cnn.mobileNetv3 import MobileNetV3
from ...imageClassificationModels.cnn.resnet import Resnet101, Resnet34
from ...imageClassificationModels.cnn.xception import Xception
from ...imageClassificationModels.vit.vit.vit import VIT
from ...imageClassificationModels.vit.swinTransformer.swinTransformer import (
    SwinTransformer,
)


class Extractors:
    """feature Extractors."""

    def mobileNetV1(model_name: str, num_classes: int):
        """_summary_.

        Args:
            num_classes (int): _description_.

        Returns:
            _type_: _description_
        """
        return MobileNetV1.prepareModel(model_name=model_name, num_classes=num_classes)

    def mobileNetV2(model_name: str, num_classes: int):
        """_summary_.

        Args:
            num_classes (int): _description_

        Returns:
            _type_: _description_
        """
        return MobileNetV2.prepareModel(model_name=model_name, num_classes=num_classes)

    def mobileNetV3(model_name: str, num_classes: int):
        """_summary_.

        Args:
            num_classes (int): _description_

        Returns:
            _type_: _description_
        """
        return MobileNetV3.prepareModel(model_name=model_name, num_classes=num_classes)

    def resnet101(model_name: str, num_classes: int):
        """_summary_.

        Args:
            num_classes (int): _description_

        Returns:
            _type_: _description_
        """
        return Resnet101.prepareModel(model_name=model_name, num_classes=num_classes)

    def resnet34(model_name: str, num_classes: int):
        """_summary_.

        Args:
            num_classes (int): _description_

        Returns:
            _type_: _description_
        """
        return Resnet34.prepareModel(model_name=model_name, num_classes=num_classes)

    def xception(model_name: str, num_classes: int):
        """_summary_.

        Args:
            num_classes (int): _description_

        Returns:
            _type_: _description_
        """
        return Xception.prepareModel(model_name=model_name, num_classes=num_classes)

    """
    def vit(model_name: str, num_classes: int):
        return VIT.prepareModel(
            model_name=model_name, num_classes=num_classes)

    def swin_t(model_name: str, num_classes: int):
        return SwinTransformer.prepareModel(
            model_name=model_name, num_classes=num_classes
        )

    def swin_s(model_name: str, num_classes: int):
        return SwinTransformer.prepareModel(
            model_name=model_name, num_classes=num_classes
        )

    def swin_b(model_name: str, num_classes: int):
        return SwinTransformer.prepareModel(
            model_name=model_name, num_classes=num_classes
        )

    def swin_l(model_name: str, num_classes: int):
        return SwinTransformer.prepareModel(
            model_name=model_name, num_classes=num_classes
        )
    """
