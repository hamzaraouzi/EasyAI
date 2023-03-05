"""a script that runs an end-to-end pipline."""
from ModelsPreparers.modelFactory import ModelFactory
from DataPreparers.dataLoaderFactory import DataLoaderFactory
from trainers.trainersFactory import TrainerFactory
import click


@click.command()
@click.option("--data_conf", type=str, help="config file for data preparation")
@click.option("--model_conf", type=str, help="config file for model preparation")
@click.option("--trainer_conf", type=str, help="config file for trainer preparation")
def main(data_conf: str, model_conf: str, trainer_conf: str):
    """runing pipeline.

    Args:
        data_conf (str): config file for data preparation.
        model_conf (str): config file for model preparation.
        trainer_conf (str): config file for trainer preparation.
    """
    data_factory = DataLoaderFactory(config_path=data_conf)
    data_preparer = data_factory()
    train_loader, val_loader, test_loader = data_preparer()

    model_preparer = ModelFactory(config_path=model_conf)
    model = model_preparer()

    trainer_factory = TrainerFactory(config_path=trainer_conf)
    trainer = trainer_factory()
    trainer(model=model, train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
