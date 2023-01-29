"""Classification training code."""
from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchmetrics.classification import (
    BinaryAccuracy,
    MulticlassAccuracy,
    BinaryPrecision,
    MulticlassPrecision,
)
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
        self.criterion = self.define_criterion()

    def define_criterion(self) -> nn.Module:
        """defining criterion.

        Returns:
            nn.Module: pytorch model.
        """
        if self.task == "binary_classification":
            return nn.BCELoss()

        elif self.task == "classification":
            return nn.CrossEntropyLoss()

    def define_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """preparing optimizer.

        Args:
            model (nn.Module): pytorch model.

        Returns:
            optim.Optimizer: optimizer.
        """
        if self.optimizer_parameters["name"] == "Adam":

            return optim.Adam(
                model.parameters(),
                lr=self.optimizer_parameters["lr"],
                betas=tuple(self.optimizer_parameters["betas"]),
                weight_decay=self.optimizer_parameters["weight_decay"],
            )

        if self.optimizer_parameters["name"] == "SGD":
            return optim.SGD(
                model.parameters(),
                lr=self.optimizer_parameters["lr"],
                momentum=self.optimizer_parameters["momentum"],
                dampening=self.optimizer_parameters["dampening"],
                nestrove=self.optimizer_parameters["nestrove"],
            )

        if self.optimizer_parameters["name"] == "RMSprop":
            return optim.RMSprop(
                model.parameters(),
                lr=self.optimizer_parameters["lr"],
                momentum=self.optimizer_parameters["momentum"],
                alpha=self.optimizer_parameters["alpha"],
                weight_decay=self.optimizer_parameters["weight_decay"],
            )

        if self.optimizer_parameters["name"] == "Adagrad":
            return optim.Adagrad(
                model.parameters(),
                lr=self.optimizer_parameters["lr"],
                lr_decay=self.optimizer_parameters["lr_decay"],
                weight_decay=self.optimizer_parameters["weight_decay"],
            )

        if self.optimizer_parameters["name"] == "Adadelta":
            return optim.Adadelta(
                model.parameters(),
                lr=self.optimizer_parameters["lr"],
                weight_decay=self.optimizer_parameters["weight_decay"],
            )

    def log_metrics(
        self,
        exp_tracker: AbstractTracker,
        y_train_true: torch.Tensor,
        y_train_pred: torch.Tensor,
        y_val_true: torch.Tensor,
        y_val_pred: torch.Tensor,
        train_loss: float,
        val_loss: float,
    ) -> dict:
        """logging metircs to experiment tracking tool.

        Args:
            exp_tracker (AbstractTracker): experiment tracking intance.
            y_train_true (torch.Tensor): groundtruth from training set.
            y_train_pred (torch.Tensor): predcitions on training set.
            y_val_true (torch.Tensor): groundtruth from validation set.
            y_val_pred (torch.Tensor): predictions from validation set.
            train_loss (float): training loss.
            val_loss (float): validation loss.

        Returns:
            dict: dictionary of classification metrics.
        """
        if self.task == "binary-classification":
            acc_fn = BinaryAccuracy()
            precision_fn = BinaryPrecision()
        else:
            num_classes = y_train_pred.shape[-1]
            acc_fn = MulticlassAccuracy(num_classes=num_classes)
            precision_fn = MulticlassPrecision(num_classes=num_classes)

        train_accuracy = acc_fn(y_train_pred, y_train_true)
        val_accuracy = acc_fn(y_val_pred, y_val_true)

        train_precision = precision_fn(y_train_pred, y_train_true)
        val_precision = precision_fn(y_val_pred, y_val_true)

        # calculate and logging (recall) other metrics
        metrics = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_precision": train_precision.item(),
            "val_precision": val_precision.item(),
        }
        exp_tracker.log_metrics(metrics=metrics)

        return metrics

    def train(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        """training function.

        Args:
            model (nn.Module): pytorch model.
            train_loader (DataLoader): training data loader.
            val_loader (DataLoader): validation data loader.
        """
        model.to(device=self.device)
        exp_tracker = self.prepare_exp_tracker()

        # TODO in config we need to log some metadata for example
        # config = {
        #    "dataset": "CIFAR10",
        #    "model": "CNN",
        #    "learning_rate": 0.01,
        #    "batch_size": 128,
        #    }
        exp_tracker.init(config=None)

        self.optimizer = self.define_optimizer(model)
        self.lr_scheduler = self.prepare_lr_scheduler()

        best_metric = (
            float("-inf") if self.monitor_metric["mode"] == "max" else float("inf")
        )
        for epoch in range(self.num_epochs):
            train_loss, y_train_true, y_train_pred = model.one_train_epoch(
                train_loader=train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                device=self.device,
            )

            val_loss, y_val_true, y_val_pred = model.one_val_epoch(
                val_loader=val_loader, criterion=self.criterion, device=self.device
            )

            metrics = self.log_metrics(
                exp_tracker=exp_tracker,
                y_train_true=y_train_true.cpu(),
                y_train_pred=y_train_pred.cpu(),
                y_val_true=y_val_true.cpu(),
                y_val_pred=y_val_pred.cpu(),
                train_loss=train_loss,
                val_loss=val_loss,
            )

            no_improvement = 0
            if (
                metrics[self.monitor_metric["name"]] > best_metric
                and self.monitor_metric["mode"] == "max"
            ) or (
                metrics[self.monitor_metric["name"]] < best_metric
                and self.monitor_metric["mode"] == "min"
            ):

                self.save_best_weights(model, model_name=model.model_name)
                best_metric = metrics[self.monitor_metric["name"]]
                no_improvement = 0

            # early stopping
            elif no_improvement < self.early_stopping:
                no_improvement += 1
            else:
                # log a message that no improvement has been made for the {no_improvement} epochs
                break

        exp_tracker.log_checkpoint()

    def run(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        """run training step.

        Args:
            model (nn.Module): pytorch model.
            train_loader (DataLoader): training data loader.
            val_loader (DataLoader): validation data loader.
        """
        self.train(model, train_loader, val_loader)
