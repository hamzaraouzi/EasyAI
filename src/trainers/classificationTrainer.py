"""Classification training code."""
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from .abstractTrainer import AbstractTrainer
from .trackers.abstractTracker import AbstractTracker
import torch


class ClassificationTrainer(AbstractTrainer):
    """ClassificationTrainer class."""

    def __init__(self, config_path: str) -> None:
        """constructor method of ClassificationTrainer class.

        Args:
            config_path (str): config file path.
        """
        super(ClassificationTrainer, self).__init__(config_path=config_path)
        # self.criterion = self.define_criterion()

    def define_criterion(self) -> nn.Module:
        """defining criterion.

        Returns:
            nn.Module: pytorch model.
        """
        if self.task == "binary_classification":
            return nn.BCELoss()

        elif self.task == "classification":
            return nn.CrossEntropyLoss()
