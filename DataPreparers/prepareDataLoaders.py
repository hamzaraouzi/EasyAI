"""This model is about DataPreparation.
"""


import yaml
import pandas as pd
from supervisedImageClassificationDataset import SupervisedImageClassicationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PrepareDataLoader:
    """DataPreparation."""

    def __init__(self, config_path: str):
        """

        Args:
            config_path (str): _description_
        """
        parameters = self.load_check_conf_file(config_path)

        self.task = parameters["task"]
        self.csv_file = parameters["csv_file"]
        self.img_dir = parameters["img_dir"]

        self.split = parameters["split"]
        self.ratios = parameters["ratios"]  # {train: ,val: ,test: } or {train: ,test: }
        self.batch_size = parameters["batch_size"]

        self.train_transfom, self.test_transform = self.prepare_transformations(
            config_path
        )

    def __create_op(self, op_info):
        """
            Loading transformation operations that has been requested by user through the config file.
        Args:
            op_info (dict): contain the name and parameters of an Albumentation operation

        Raises:
            Not implemented exception in case of an operation that not yet implemented or an operation that dosen't exist at all

        Returns:
            operation : the disared operation parameters with disered paramters
        """

        op_name = list(op_info.keys())[0]
        if op_name == "resize":
            return A.Resize(
                height=op_info["height"], width=op_info["width"], p=op_info["p"]
            )

        if op_name == "horizontalFlip":
            return A.HorizontalFlip(p=op_info["p"])

        if op_name == "verticalFlip":
            return A.VerticalFlip(p=op_info["p"])

        if op_name == "centralCrop":
            return A.CenterCrop(
                height=op_info["height"], width=op_info["width"], p=op_info["p"]
            )

        if op_name == "rotate":
            return A.Rotate(
                limit=[op_info["minAngle"], op_info["maxAngle"]], p=op_info["p"]
            )

        if op_name == "normalize":

            return A.Normalize(
                mean=tuple(op_info["normalize"]["mean"]),
                std=tuple(op_info["normalize"]["std"]),
            )

        raise f"{op_name} is either not yet implmented or dosen't exist in Albumentation"

    def prepare_transformations(self, config_path: str):
        """This function reads the config yaml file and
            prepare the augmentation trasnform with albumentation

        Args:
            config_path (str): path to the configuration file

        Returns:
            train_transform: an albumantation transform for training set
            test_transform: an albumantation transform for test set
        """
        parameters = self.load_check_conf_file(config_path)
        train_compose = []
        for op_info in parameters["train_transforms"]:
            op = self.__create_op(op_info)
            train_compose.append(op)

        test_compose = []
        for op_info in parameters["test_transforms"]:
            op = self.__create_op(op_info)
            test_compose.append(op)

        train_compose.append(ToTensorV2())
        test_compose.append(ToTensorV2())

        return A.Compose(train_compose), A.Compose(test_compose)

    def __call__(self):
        """Data Preparation

        Returns:
            DataLoaders
        """
        # TODO handel ratios and diffrents splits
        df = pd.read_csv(self.csv_file)

        train_df, test_df = train_test_split(df, test_size=self.ratios["test"])
        if self.split == "train-test":

            train_ds = SupervisedImageClassicationDataset(
                img_dir=self.img_dir,
                df=train_df,
                target_column=self.target_column,
                transform=self.train_transfom,
            )

            test_ds = SupervisedImageClassicationDataset(
                img_dir=self.img_dir,
                df=test_df,
                target_column=self.target_column,
                transform=self.test_transfom,
            )

            train_loader = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            test_loader = DataLoader(
                test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2
            )
            return train_loader, test_loader

        if self.split == "train-val-test":

            val_df, test_df = train_test_split(
                test_df,
                test_size=self.ratios["test"]
                / (self.ratios["test"] + self.ratios["val"]),
            )

            train_ds = SupervisedImageClassicationDataset(
                img_dir=self.img_dir,
                df=train_df,
                target_column=self.target_column,
                transform=self.train_transfom,
            )

            test_ds = SupervisedImageClassicationDataset(
                img_dir=self.img_dir,
                df=test_df,
                target_column=self.target_column,
                transform=self.test_transfom,
            )

            val_ds = SupervisedImageClassicationDataset(
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

    def load_check_conf_file(self, config_path: str):
        """Loading and checking config file for DataLoader

        Args:
            config_path (str): path to the yaml config file

        Returns:
            dict: key->values
        """
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)

        params2values = {}
        for d in conf_values["Dataset"]:
            for k, v in zip(d.keys(), d.values()):
                params2values[k] = v

        return params2values
