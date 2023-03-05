"""Trainers Factory."""
import yaml
from .abstractTrainer import AbstractTrainer
from .classificationTrainer import ClassificationTrainer
from .segmentationTrainer import SegmentationTrainer


class TrainerFactory:
    """TrainerFactory class."""

    def __init__(self, config_path: str) -> None:
        """init method for trainer factory.

        Args:
            config_path (str): _description_
        """
        param2values = self.load_check_conf_file(config_path=config_path)
        self.config_path = config_path
        self.task = param2values["task"]

    def prepareTrainer(self) -> AbstractTrainer:
        """prepareTrainer method.

        Returns:
            AbstractTrainer: desired trainer.
        """
        if self.task == "classification":
            return ClassificationTrainer(config_path=self.config_path)

        if self.task in [
            "multiclass-semantic-segmentation",
            "binary-semantic-segmentation",
        ]:
            return SegmentationTrainer(config_path=self.config_path)

    def load_check_conf_file(self, config_path: str):
        """method for loading the configuration from a yaml file.

        Args:
            config_path (str): config file path.

        Returns:
            dict: dictionary that maps parameter to values.
        """
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)

        params2values = {}
        for d in conf_values["training"]:
            for k, v in zip(d.keys(), d.values()):
                if k != "optimizer":
                    params2values[k] = v

        return params2values

    def __call__(self) -> AbstractTrainer:
        """call method for trainer prepartion.

        Returns:
            AbstractTrainer: desired trainer.
        """
        return self.prepareTrainer()
