"""Pytorch Dataset class for Image Classification."""

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import numpy as np


class ImageClassicationDataset(Dataset):
    """SupervisedImageClassificationData pytorch Dataset class."""

    def __init__(self, img_dir: str, df: pd.DataFrame, transform) -> None:
        """Init method for SupervisedImageClassificationData.

        Args:
            img_dir (str): images folder
            df (pd.DataFrame): df frame for metadata and labels
            transform (_type_): Augmentation operations
        """
        self.df = df
        self.df.reset_index(inplace=True)

        self.target_column = "class"

        self.classes = {cls: i for i, cls in enumerate(set(df[self.target_column]))}
        self.num_classes = len(self.classes)

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Returning the number of elements of the dataset.

        Returns:
            int: number of examples.
        """
        return len(self.df)

    def __getitem__(self, idx: int):
        """get element from the data.

        Args:
            idx (int): _description_

        Returns:
            image: torch.Tensor
            y : torch.Tensor
        """
        y = self.classes[self.df.loc[idx, self.target_column]]

        image_name = self.df.loc[idx, "image_name"]

        img_path = os.path.join(self.img_dir, image_name)
        img = np.array(Image.open(img_path).convert("RGB"))

        img = self.transform(image=img) if self.transform is not None else img

        return img["image"], y
