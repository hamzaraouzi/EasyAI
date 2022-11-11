"""Pytorch Dataset class for Image Classification.
"""


from typing import Optional
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd


class SupervisedImageClassicationDataset(Dataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, transform):

        self.df = df
        self.df.reset_index(inplace=True)

        self.target_column = "class"

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        y = self.df.loc[idx, self.target_column]
        image_name = self.df.loc[idx, "image_name"]

        img_path = os.path.join(self.img_dir, image_name)
        img = Image.open(img_path).convert("RGB")

        img = self.transform(img) if self.transform is not None else img
        return img, torch.tensor(y)
