"""Abstract Image Classifier classifier class.

Its class from which all Image classification models , will Inherent some methods.
"""

from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Literal, Tuple, List
from torchmetrics.classification import (
    BinaryAccuracy,
    MulticlassAccuracy,
    BinaryPrecision,
    MulticlassPrecision,
)


class AbstractClassifier(nn.Module):
    """An Abstract Image classication class."""

    def __init__(self, model_name: str, num_classes: int) -> None:
        """init method for AbstractClassifier.

        Args:
            model_name (str): _description_
            num_classes (int): _description_
        """
        super(AbstractClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.task = "classification" if num_classes > 1 else "binary-classification"

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """features extraction method.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            List[torch.Tensor]: list of features.
        """
        pass

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
        model_name: str, in_channels: int = 3, num_classes: int = 10, **kwargs: dict
    ) -> nn.Module:
        """Desired model preparation.

        Args:
            in_channels (int): _description_. Defaults to 3.
            num_classes (int): _description_. Defaults to 10.
            kwargs (dict): _description_.

        Returns:
            nn.Module: _description_
        """
        pass

    def prepare_pred_examples(
        self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor, n_samples: int
    ) -> dict:
        """_summary_.

        Args:
            X (torch.Tensor): _description_
            y (torch.Tensor): _description_
            y_pred (torch.Tensor): _description_
            n_samples (int): _description_

        Returns:
            dict: _description_
        """
        # TODO: implement this method.
        return None

    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> Tuple[float, float]:
        """logging metircs to experiment tracking tool.

        Args:
            predictions (torch.Tensor): models predictions.
            targets (torch.Tensor): targets predictions.
            device (Literal['cuda', 'cpu'], optional): training hardware. Defaults to 'cuda'.

        Returns:
            Tuple[float, float]: _description_
        """
        if self.task == "binary-classification":
            acc_fn = BinaryAccuracy().to(device=device)
            precision_fn = BinaryPrecision().to(device=device)
        else:
            num_classes = predictions.shape[-1]
            acc_fn = MulticlassAccuracy(num_classes=num_classes).to(device=device)
            precision_fn = MulticlassPrecision(num_classes=num_classes).to(
                device=device
            )

        accuracy = acc_fn(predictions, targets)
        precision = precision_fn(predictions, targets)

        return accuracy.item(), precision.item()

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
            dict: validation loss, and validation metrics.
        """
        self.eval()
        val_loss, mean_acc, mean_prec = 0, 0, 0

        with torch.no_grad():
            for _, (X, y) in enumerate(tqdm(val_loader, leave=True)):
                X, y = X.to(device), y.to(device)
                y_pred = self(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item() / len(val_loader)

                acc, prec = self.calculate_metrics(
                    predictions=y_pred, targets=y, device=device
                )
                mean_acc += acc / len(val_loader)
                mean_prec += prec / len(val_loader)

        pred_examples = self.prepare_pred_examples(X=X, y=y, y_pred=y_pred, n_samples=8)
        return {
            "val_loss": val_loss,
            "val_accuracy": mean_acc,
            "val_precision": mean_prec,
        }, pred_examples

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
            dict: mean loss, all true labels, all model's predictions.
        """
        self.train()
        train_loss, mean_acc, mean_prec = 0, 0, 0

        for _, (X, y) in enumerate(tqdm(train_loader, leave=True)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = self(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)

            acc, prec = self.calculate_metrics(
                predictions=y_pred, targets=y, device=device
            )
            mean_acc += acc / len(train_loader)
            mean_prec += prec / len(train_loader)

        if scheduler is not None:
            scheduler.step()

        pred_examples = self.prepare_pred_examples(X=X, y=y, y_pred=y_pred, n_samples=8)
        return {
            "train_loss": train_loss,
            "train_accuracy": mean_acc,
            "train_precision": mean_prec,
        }, pred_examples
