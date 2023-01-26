"""Abstract Trainer."""
from typing import Optional, Union
from abc import abstractmethod
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from .trackers.abstractTracker import AbstractTracker
import torch
import os
from .trackers.wandbTracker import WandBTracker
from torch.optim.lr_scheduler import (
    _LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CyclicLR,
    ReduceLROnPlateau,
)
from torch.optim import Optimizer


class AbstractTrainer:
    """AbstractTrainer class."""

    def __init__(self, config_path: str) -> None:
        """constructor method of AbstractTrainer class.

        Args:
            config_path (str): config file path.
        """
        self.config_path = config_path
        params2values, self.optimizer_parameters = self.load_check_conf_file(
            config_path
        )
        self.task = params2values["task"]
        self.device = params2values["device"]
        self.num_epochs = params2values["num_epochs"]
        self.early_stopping = params2values["earlystoping_after"]
        self.project = params2values["project"]
        self.experiment_tracker = params2values["experiment_tracker"]
        self.monitor_metric = params2values["monitor_metric"]

        self.lr_schedular_conf = (
            params2values["learning_rate_schedular"]
            if "learning_rate_schedular" in params2values.keys()
            else None
        )

        self.optimizer = None  # it

    def load_check_conf_file(self, config_path: str):
        """method for loading the configuration from a yaml file.

        Args:
            config_path (str): config file path.

        Returns:
            dict: dictionary that maps parameter to values.
            dict: optimizer parameters.
        """
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)

        params2values = {}
        optimizer_parameters = {}

        for d in conf_values["training"]:
            for k, v in zip(d.keys(), d.values()):
                if k != "optimizer":
                    params2values[k] = v

                else:
                    for dd in v:
                        for kk, vv in zip(dd.keys(), dd.values()):
                            if kk != "optimizer":
                                optimizer_parameters[kk] = vv

        return params2values, optimizer_parameters

    def prepare_exp_tracker(self) -> AbstractTracker:
        """preparing experiment tracker.

        Returns:
            AbstractTracker: _description_
        """
        if self.experiment_tracker["name"] == "wandb":
            return WandBTracker(
                project=self.project, tracking_conf=self.experiment_tracker
            )

    def prepare_lr_schedular(self) -> Union[Optimizer, _LRScheduler]:
        """prepare learning rate schedulars.

        Returns:
            Union[Optimizer, _LRScheduler]: the optimizer or the learning rate schedular.
        """
        if self.lr_schedular_conf is None:
            return self.optimizer

        if self.lr_schedular_conf["name"] == "stepLR":
            return StepLR(
                optimizer=self.optimizer,
                step_size=self.lr_schedular_conf["step_size"],
                verbose=True,
            )

        if self.lr_schedular_conf["name"] == "multistepLR":
            return MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.lr_schedular_conf["milestones"],
                gamma=self.lr_schedular_conf["gamma"],
                verbose=True,
            )

        if self.lr_schedular_conf["name"] == "exponentialLR":
            return ExponentialLR(
                optimizer=self.optimizer,
                gamma=self.lr_schedular_conf["gamma"],
                verbose=True,
            )

        if self.lr_schedular_conf["name"] == "cyclicalLR":
            return CyclicLR(
                optimizer=self.optimizer,
                base_lr=self.lr_schedular_conf["base_lr"],
                max_lr=self.lr_schedular_conf["max_lr"],
                step_size_up=self.lr_schedular_conf["step_size_up"],
                mode=self.lr_schedular_conf["mode"],
                verbose=True,
            )
        if self.lr_schedular_conf["name"] == "reduceLROnPlateau":
            return ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=self.lr_schedular_conf["factor"],
                patience=self.lr_schedular_conf["patience"],
            )

    def save_best_weights(self, model: nn.Module, model_name: str) -> None:
        """save best weights.

        Args:
            model (nn.Module): pytorch model.
            model_name (str):  model name.
        """
        os.makedirs("../checkpoints", exist_ok=True)
        torch.save(model, f"../checkpoints/{model_name}.pth")

    @abstractmethod
    def define_criterion(self):
        """defining criterion."""
        pass

    @abstractmethod
    def define_optimizer(self, model: nn.Module):
        """define optimizer.

        Args:
            model (nn.Module): pytorch model.
        """
        pass

    @abstractmethod
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
        """training function.

        Args:
            model (nn.Module): pytorch model.
            train_loader (DataLoader): training data loader.
            val_loader (DataLoader): validation data loader.
        """
        pass

    @abstractmethod
    def run(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
    ):
        """run training step.

        Args:
            model (nn.Module): pytorch model.
            train_loader (DataLoader): training data loader.
            val_loader (DataLoader): validation data loader.
        """
        pass

    @abstractmethod
    def log_metrics(
        self,
        exp_tracker: AbstractTracker,
        y_train_true: torch.Tensor,
        y_train_pred: torch.Tensor,
        y_val_true: torch.Tensor,
        y_val_pred: torch.Tensor,
        train_loss: torch.Tensor,
        val_loss: torch.Tensor,
    ) -> None:
        """logging metircs to experiment tracking tool.

        Args:
            exp_tracker (AbstractTracker): experiment tracking intance.
            y_train_true (torch.Tensor): groundtruth from training set.
            y_train_pred (torch.Tensor): predcitions on training set.
            y_val_true (torch.Tensor): groundtruth from validation set.
            y_val_pred (torch.Tensor): predictions from validation set.
            train_loss (torch.Tensor): training loss.
            val_loss (torch.Tensor): validation loss.
        """
        pass

    def log_checkpoint(self, ckpt_path: str = "../checkpoints/*"):
        """log best weights to experiment tracker.

        Args:
            ckpt_path (str): _description_. Defaults to "../checkpoints".
        """
        pass

    def __call__(
        self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        """Call method for trainers.

        Args:
            model (nn.Module): _description_
            train_loader (DataLoader): _description_
            val_loader (DataLoader): _description_
        """
        self.run(model, train_loader, val_loader)
