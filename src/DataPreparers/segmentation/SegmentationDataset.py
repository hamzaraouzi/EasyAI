"""semantic Segmentation dataset."""
from torch.utils.data import Dataset
import os
from typing import Tuple, Optional, Literal
import torch
from PIL import Image
import albumentations as A
import numpy as np


class SegmentationDataset(Dataset):
    """Segmentation dataset."""

    def __init__(
        self,
        dataset_path: str,
        task: Literal[
            "binary-semantic-segmentation", "multiclass-semantic-segmentation"
        ],
        subset: Literal["train", "valid", "test"],
        transform: Optional[A.Compose],
    ) -> None:
        """pyotrch Dataset class for semantic segmentation dataset.

        Args:
            dataset_path (str): _description_
            task (Literal[ &quot;binary): _description_
            subset (Literal[&quot;train&quot;, &quot;valid&quot;, &quot;test&quot;]): _description_
            transform (Optional[A.Compose]): _description_
        """
        self.masks_dir = os.path.join(dataset_path, "masks", subset)
        self.images_dir = os.path.join(dataset_path, "images", subset)

        self.images_names = sorted(os.listdir(self.images_dir))
        self.masks_names = sorted(os.listdir(self.masks_dir))

        self.scale = 1 if task == "multiclass-semantic-segmentation" else 255
        self.transform = transform

    def __len__(self) -> int:
        """get the size of the dataset.

        Returns:
            int: the size of the dataset.
        """
        return len(self.images_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """get item from the dataset.

        Args:
            index (int): id of the item.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: image tensor, mask tensor.
        """
        image = np.array(
            Image.open(os.path.join(self.images_dir, self.images_names[index])).convert(
                "RGB"
            )
        )
        mask = np.array(
            Image.open(os.path.join(self.masks_dir, self.masks_names[index])).convert(
                "L"
            ),
            dtype=np.float32,
        )

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask /= self.scale

        return image.float(), mask.float()
