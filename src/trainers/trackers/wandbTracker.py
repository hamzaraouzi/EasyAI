"""wights and biases tracker."""
from .abstractTracker import AbstractTracker
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
        self.artifact = wandb.Artifact(name="model-artifact", type="model")

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

    def log_checkpoint(self, ckpt_path: str = "../checkpoints/*"):
        """best weights to weights and biases.

        Args:
            ckpt_path (str): _description_. Defaults to "../checkpoints".
        """
        self.artifact.add_dir("../checkpoints")
        wandb.log_artifact(self.artifact)

    def __call__(self, metrics: dict) -> None:
        """logging metrics to wandb.

        Args:
            metrics (dict): dictionary of metrics.
        """
        self.log_metrics(metrics)
