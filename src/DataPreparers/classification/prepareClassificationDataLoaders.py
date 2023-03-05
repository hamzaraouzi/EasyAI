"""This model is about DataPreparation for image classifications."""


import yaml
import pandas as pd
from .ImageClassificationDataset import ImageClassicationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from ..abstractDataLoaderPreparer import AbstractDataPreparer


class ClassificaionDataLoader(AbstractDataPreparer):
    """DataPreparation."""

    def __init__(self, config_path: str):
        """__init__ method for DataLoader class.

        Args:
            config_path (str): _description_
        """
        super(ClassificaionDataLoader, self).__init__(config_path)

        self.task = self.parameters["task"]
        self.csv_file = self.parameters["csv_file"]
        self.img_dir = self.parameters["img_dir"]

        self.split = self.parameters["split"]
        self.ratios = self.parameters[
            "ratios"
        ]  # {train: ,val: ,test: } or {train: ,test: }
        self.batch_size = self.parameters["batch_size"]

    def __call__(self):
        """Data Preparation.

        Returns:
            DataLoaders
        """
        # TODO handel ratios and diffrents splits
        df = pd.read_csv(self.csv_file)

        train_df, test_df = train_test_split(df, test_size=self.ratios["test"])
        if self.split == "train-test":

            train_ds = ImageClassicationDataset(
                img_dir=self.img_dir,
                df=train_df,
                transform=self.train_transfom,
            )

            test_ds = ImageClassicationDataset(
                img_dir=self.img_dir,
                df=test_df,
                transform=self.test_transform,
            )

            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            test_loader = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            return train_loader, test_loader, None

        if self.split == "train-val-test":

            val_df, test_df = train_test_split(
                test_df,
                test_size=self.ratios["test"]
                / (self.ratios["test"] + self.ratios["val"]),
            )

            train_ds = ImageClassicationDataset(
                img_dir=self.img_dir,
                df=train_df,
                target_column=self.target_column,
                transform=self.train_transfom,
            )

            test_ds = ImageClassicationDataset(
                img_dir=self.img_dir,
                df=test_df,
                target_column=self.target_column,
                transform=self.test_transfom,
            )

            val_ds = ImageClassicationDataset(
                img_dir=self.img_dir,
                df=val_df,
                target_column=self.target_column,
                transform=self.test_transfom,
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

            return train_loader, val_loader, test_loader
