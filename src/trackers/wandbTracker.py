"""wights and biases tracker."""
from abstractTracker import AbstractTracker
import wandb


class WandBTracker(AbstractTracker):
    """WandBTracker class."""

    def __init__(self, project: str, tracking_conf: dict) -> None:
        """constructor of WandBTrackerClass.

        Args:
            project (str): project name.
            tracking_conf (dict): tracking configuraations.
        """
        self.tracking_conf = tracking_conf
        self.name = tracking_conf["name"]
        self.credentials = tracking_conf["credentials"]
        self.project = project

    def init(self, config: dict):
        """initialze wandb tracking.

        Args:
            config (dict): _description_
        """
        wandb.login(key=self.credentials["key"])
        wandb.init(project=self.project, config=config)

    def log_metrics(self, metrics: dict) -> None:
        """log metrics.

        Args:
            metrics (dict): metrics
        """
        wandb.log(metrics)

    def __call__(self, metrics: dict) -> None:
        """logging metrics to wandb.

        Args:
            metrics (dict): _description_
        """
        self.log_metrics(metrics)
