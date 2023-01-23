"""a script that runs an end-to-end pipline."""
from ModelsPreparers.ClassificationModel import ClassificationModel
from DataPreparers.prepareDataLoaders import PrepareDataLoader
from trainers.classificationTrainer import ClassificationTrainer
import click


@click.command()
@click.option("--data_conf", help="config file for data preparation")
@click.option("--model_conf", help="config file for model preparation")
@click.option("--trainer_conf", help="config file for trainer preparation")
def main(data_conf: str, model_conf: str, trainer_conf: str):
    """runing pipeline.

    Args:
        data_conf (str): config file for data preparation.
        model_conf (str): config file for model preparation.
        trainer_conf (str): config file for trainer preparation.
    """
    data_preparer = PrepareDataLoader(config_path=data_conf)
    train_loader, val_loader, test_loader = data_preparer()

    # TODO needs to be implemented this way:
    # model preparer that prepare classification as well as segmentation models and object detection models.
    model_preparer = ClassificationModel(config_path=model_conf)
    model = model_preparer()

    trainer = ClassificationTrainer(config_path=trainer_conf)
    trainer(model=model, train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
