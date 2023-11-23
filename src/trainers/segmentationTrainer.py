"""Semantic segmentation Trainer."""
from .abstractTrainer import AbstractTrainer
import torch.nn as nn


class SegmentationTrainer(AbstractTrainer):
    """Segmentation Trainer."""

    def __init__(self, config_path: str) -> None:
        """Segmentation Trainer class constructor.

        Args:
            config_path (str): config path.
        """
        super().__init__(config_path)
        # self.criterion = self.define_criterion()

    def define_criterion(self) -> nn.Module:
        """defining criterion.

        Returns:
            nn.Module: pytorch model.
        """
        # TODO I need implemtations of semantic segmentation loss functions.
        if self.task == "multiclass-semantic-segmentation":
            return nn.CrossEntropyLoss()

        if self.task == "binary-semantic-segmentation":
            if self.criterion == "BceLoss":
                return nn.BCEWithLogitsLoss()
            elif self.criterion == "NLLoss2d":
                return nn.NLLLoss2d()
