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
from torch import optim
from collections import ChainMap


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
            dict(ChainMap(*params2values["learning_rate_scheduler"]))
            if "learning_rate_scheduler" in params2values.keys()
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

    def prepare_lr_scheduler(self) -> _LRScheduler:
        """prepare learning rate schedulars.

        Returns:
            _LRScheduler: the optimizer or the learning rate scheduler.
        """
        if self.lr_schedular_conf is None:
            return None

        if self.lr_schedular_conf["name"] == "stepLR":
            return StepLR(
                optimizer=self.optimizer,
                step_size=self.lr_schedular_conf["step_size"],
                gamma=self.lr_schedular_conf["gamma"],
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
                verbose=True,
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
    def define_criterion(self) -> nn.Module:
        """defining criterion.

        Returns:
            nn.Module: pytorch model.
        """
        pass

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

    @abstractmethod
    def log_metrics(self, exp_tracker: AbstractTracker, metrics: dict) -> None:
        """logging metircs to experiment tracking tool.

        Args:
            exp_tracker (AbstractTracker): experiment tracking instance.
            metrics (dict): training and validation metrics.
        """
        exp_tracker.log_metrics(metrics=metrics)

    def log_checkpoint(self, ckpt_path: str = "../checkpoints/*"):
        """log best weights to experiment tracker.

        Args:
            ckpt_path (str): _description_. Defaults to "../checkpoints".
        """
        pass

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
        self.exp_tracker = self.prepare_exp_tracker()

        # TODO in config we need to log some metadata for example
        # config = {
        #    "dataset": "CIFAR10",
        #    "model": "CNN",
        #    "learning_rate": 0.01,
        #    "batch_size": 128,
        #    }
        self.exp_tracker.init(config=None)

        self.criterion = self.define_criterion()

        self.optimizer = self.define_optimizer(model)
        self.lr_scheduler = self.prepare_lr_scheduler()

        best_metric = (
            float("-inf") if self.monitor_metric["mode"] == "max" else float("inf")
        )
        for epoch in range(self.num_epochs):
            metrics = model.one_train_epoch(
                train_loader=train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                device=self.device,
            )

            val_metrics = model.one_val_epoch(
                val_loader=val_loader, criterion=self.criterion, device=self.device
            )

            metrics.update(val_metrics)

            self.log_metrics(exp_tracker=self.exp_tracker, metrics=metrics)

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

        self.exp_tracker.log_checkpoint()

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
