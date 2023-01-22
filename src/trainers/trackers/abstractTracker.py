"""abstract experiment tracking class."""
from abc import abstractmethod


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
