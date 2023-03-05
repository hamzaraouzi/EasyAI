"""data laoders factory class."""
import yaml
from .abstractDataLoaderPreparer import AbstractDataPreparer
from .classification.prepareClassificationDataLoaders import ClassificaionDataLoader
from .segmentation.prepareSegmentationDataLoader import SegmentationDataLoader


class DataLoaderFactory:
    """DataLoaderFactory class."""

    def __init__(self, config_path: str) -> None:
        """data fctory init method.

        Args:
            config_path (str): path to configuration file.
        """
        parameters = self.load_check_conf_file(config_path)
        self.config_path = config_path
        self.task = parameters["task"]

    def prepareDataLoader(self) -> AbstractDataPreparer:
        """prepare segmentation dataloader.

        Returns:
            AbstractDataPreparer: DataLoaders.
        """
        if self.task == "classification":
            return ClassificaionDataLoader(config_path=self.config_path)

        elif self.task in [
            "multiclass-semantic-segmentation",
            "binary-semantic-segmentation",
        ]:
            return SegmentationDataLoader(config_path=self.config_path)

    def load_check_conf_file(self, config_path: str) -> dict:
        """Loading and checking config file for DataLoader.

        Args:
            config_path (str): path to the yaml config file

        Returns:
            dict: key->values
        """
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)

        params2values = {}
        for d in conf_values["Dataset"]:
            for k, v in zip(d.keys(), d.values()):
                params2values[k] = v

        return params2values

    def __call__(self) -> AbstractDataPreparer:
        """prepare data loader.

        Returns:
            AbstractDataPreparer: desired data loader.
        """
        return self.prepareDataLoader()
