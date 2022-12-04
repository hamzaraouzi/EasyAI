"""Classification Model is a class that will manage loading and preparation of desired model."""
import torch.nn as nn
import yaml

from .imageClassificationModels.abstractClassifier import AbstractClassifier


class ClassificationModel:
    """Classification Model is a class that will manage loading and preparation of desired model."""

    def __init__(self, config_path: str) -> None:
        """Init method for classificationModel class.

        Args:
            config_path (str): the path to the yaml config path
        """
        super(ClassificationModel, self).__init__()
        params2values = self.load_check_conf_file(config_path)

        self.model_name = params2values["name"]
        self.task = params2values["task"]
        self.pretrained = params2values["pretrained"]
        self.num_classes = params2values["num_classes"]

    def load_check_conf_file(self, config_path: str) -> dict:
        """Loading desired model configuration from  yaml file.

        Args:
            config_path (str): the path to the yaml config path.

        Returns:
            dict: _description_
        """
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)

        params2values = {}
        for d in conf_values["model"]:
            for key, values in zip(d.keys(), d.values()):
                params2values[key] = values

        return params2values
