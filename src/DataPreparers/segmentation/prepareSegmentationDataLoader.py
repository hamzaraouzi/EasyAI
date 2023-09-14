"""This model is about DataPreparation."""
import pandas as pd
from .SegmentationDataset import SegmentationDataset
from torch.utils.data import DataLoader
from ..abstractDataLoaderPreparer import AbstractDataPreparer


class SegmentationDataLoader(AbstractDataPreparer):
    """DataPreparation."""

    def __init__(self, config_path: str):
        """__init__ method for DataLoader class.

        Args:
            config_path (str): _description_
        """
        super(SegmentationDataLoader, self).__init__(config_path)

        self.task = self.parameters["task"]
        self.dataset_path = self.parameters["dataset_path"]
        self.batch_size = self.parameters["batch_size"]
        self.split = self.parameters["split"]

    def __call__(self):
        """Data Preparation.

        Returns:
            DataLoaders
        """
        if self.split == "train-test":

            train_ds = SegmentationDataset(
                subset="train",
                task=self.task,
                dataset_path=self.dataset_path,
                transform=self.train_transfom,
            )

            test_ds = SegmentationDataset(
                subset="test",
                task=self.task,
                dataset_path=self.dataset_path,
                transform=self.train_transfom,
            )

            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            test_loader = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            return train_loader, test_loader, None

        if self.split == "train-val-test":

            train_ds = SegmentationDataset(
                subset="train",
                task=self.task,
                dataset_path=self.dataset_path,
                transform=self.train_transfom,
            )

            test_ds = SegmentationDataset(
                subset="test",
                task=self.task,
                dataset_path=self.dataset_path,
                transform=self.train_transfom,
            )

            val_ds = SegmentationDataset(
                subset="valid",
                task=self.task,
                dataset_path=self.dataset_path,
                transform=self.train_transfom,
            )

            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            test_loader = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )

            # additional loader will be used just for for quantization
            # will be ignored in case of optimization is applied

            calib_quantization_loader = DataLoader(
                train_ds, batch_size=1, shuffle=False
            )
            valid_quantization_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
            return (
                train_loader,
                val_loader,
                test_loader,
                calib_quantization_loader,
                valid_quantization_loader,
            )
