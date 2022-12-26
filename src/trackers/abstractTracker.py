"""abstract experiment tracking class."""
from abc import abstractmethod
from wandbTracker import WandBTracker


class AbstractTracker:
    """Abstract tracking class."""

    @abstractmethod
    def init(self, config: dict):
        """experiment tracking initialization.

        Args:
            config (dict): configurations.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict):
        """logging metrics to experiment tracking tool.

        Args:
            metrics (dict): metrcs.
        """
        pass

    @staticmethod
    def prepareTracker(name: str, project: str, tracking_conf: dict):
        """preparation of the experiment tracking.

        Args:
            name (str): _description_
            project (str): _description_
            tracking_conf (dict): _description_

        Returns:
            _type_: _description_
        """
        if name == "wandb":
            return WandBTracker(project=project, tracking_conf=tracking_conf)
