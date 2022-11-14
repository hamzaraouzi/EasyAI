"""Abstract Image Classifier classifier class.

Its class from which all Image classification models , will Inherent some methods.
"""

from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Literal, Tuple


class AbstractClassifier(nn.Module):
    """An Abstract Image classication class."""

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
        model_name: str, in_channels: int = 3, num_classes: int = 10
    ) -> nn.Module:
        """Desired model paration.

        Args:
            in_channels (int): input channels.
            num_classes (int): _description_

        Returns:
            nn.Module: _description_
        """
        pass

    def one_val_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Validation pass.

        Args:
            val_loader (DataLoader): validation data loader.
            criterion (nn.Module): loss function.
            device (Literal['cuda', 'cpu'], optional): training hardware. Defaults to 'cuda'.

        Returns:
            Tuple[float, torch.Tensor, torch.Tensor]: mean loss, all true labels, all model's predictions.
        """
        self.eval()
        val_loss = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for _, (X, y) in enumerate(tqdm(val_loader, leave=True)):
                X, y = X.to(device), y.to(device)
                y_pred = self(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item() / len(val_loader)

                all_pred.append(y_pred)
                all_true.append(y)

            return val_loss, torch.cat(all_true), torch.cat(all_pred)

    def one_train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Training epoch.

        Args:
            train_loader (DataLoader): _description_
            criterion (nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            device (Literal['cuda', 'cpu'], optional): trainig hardware. Defaults to "cuda".

        Returns:
            Tuple[float, torch.Tensor, torch.Tensor]: mean loss, all true labels, all model's predictions.
        """
        self.train()
        train_loss = 0
        all_pred, all_true = [], []

        for _, (X, y) in enumerate(tqdm(train_loader, leave=True)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = self(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_loader)
            all_pred.append(y_pred)
            all_true.append(y)

        return train_loss, torch.cat(all_true), torch.cat(all_pred)
