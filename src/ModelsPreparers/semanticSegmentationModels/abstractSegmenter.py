"""abstract semantic segmentation model."""
from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Literal, Tuple
from torchmetrics.functional import dice, jaccard_index


class AbstrctSegmenter(nn.Module):
    """An Abstract semantic segmentation model class."""

    def __init__(self, in_channels: int, num_classes: int, model_name: str):
        """init method for abstract segmenter.

        Args:
            in_channels (int): input channels.
            num_classes (int): number of classes.
            model_name (str): model_name.
        """
        super(AbstrctSegmenter, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass method.

        It's a methode inhereted from nn.Module class,
        that define the the sequence on computing the output for a batch of examples.

        Args:
            x (torch.Tensor): batch of examples

        Returns:
            torch.Tensor: the output of the forward pass
        """
        pass

    @abstractmethod
    def prepareModel(
        self, model_name: str, in_channels: int = 3, num_classes: int = 10
    ) -> nn.Module:
        """Desired model preparation.

        Args:
            model_name (str): model_name.
            in_channels (int): input channels.
            num_classes (int): number of classes.

        Returns:
            nn.Module: _description_
        """
        pass

    def calculate_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float]:
        """calculate semantic segmentation metrics.

        Args:
            predictions (torch.Tensor): model predictions.
            targets (torch.Tensor): groundtruth segmentations.

        Returns:
            Tuple[float, float]: dice and iou scores
        """
        # TODO: fix mertics calculation for segmentation.
        task = (
            "multiclass-semantic-segmentation"
            if self.num_classes > 1
            else "binary-sematic-segmentation"
        )
        predictions = (
            torch.argmax(torch.softmax(predictions, dim=1), dim=1)
            if self.num_classes > 1
            else torch.sigmoid(predictions)
        )
        dice_score = dice(
            preds=predictions,
            target=targets.int(),
            # num_classes=self.num_classes,
        )

        task = task.split("-")[0]  # binary or multiclass
        iou_score = jaccard_index(
            preds=predictions,
            target=targets.int(),
            task=task,
            num_classes=self.num_classes,
        )

        return dice_score.item(), iou_score.item()

    def one_val_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> dict:
        """Validation pass.

        Args:
            val_loader (DataLoader): validation data loader.
            criterion (nn.Module): loss function.
            device (Literal['cuda', 'cpu'], optional): training hardware. Defaults to 'cuda'.

        Returns:
            dict: mean loss, validation metrics.
        """
        self.eval()
        val_loss, mean_val_dice, mean_val_iou = 0, 0, 0

        with torch.no_grad():
            for _, (X, y) in enumerate(tqdm(val_loader, leave=True)):
                X, y = X.to(device), y.to(device)
                y_pred = self(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item() / len(val_loader)

                dice_score, iou_score = self.calculate_metrics(
                    predictions=y_pred, targets=y
                )
            mean_val_dice += dice_score / len(val_loader)
            mean_val_iou += iou_score / len(val_loader)

        return {
            "val_loss": val_loss,
            "val_dice_score": mean_val_dice,
            "val_iou_score": mean_val_iou,
        }

    def one_train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> dict:
        """Training epoch.

        Args:
            train_loader (DataLoader): _description_
            criterion (nn.Module): _description_
            optimizer (Union[Optimizer, _LRScheduler]): the optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler.
            device (Literal['cuda', 'cpu'], optional): trainig hardware. Defaults to "cuda".

        Returns:
            dict: loss and training metrics.
        """
        self.train()
        train_loss, mean_train_dice, mean_train_iou = 0, 0, 0

        for _, (X, y) in enumerate(tqdm(train_loader, leave=True)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = self(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

            dice_score, iou_score = self.calculate_metrics(
                predictions=y_pred, targets=y
            )
            mean_train_dice += dice_score / len(train_loader)
            mean_train_iou += iou_score / len(train_loader)

        if scheduler is not None:
            scheduler.step()
        return {
            "train_loss": train_loss,
            "train_dice_score": mean_train_dice,
            "train_iou_score": mean_train_iou,
        }
