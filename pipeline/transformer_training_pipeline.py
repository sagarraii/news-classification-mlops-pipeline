import sys

from src.config.configuration import ConfigurationManager
from src.components.transformer_trainer import TransformerTrainer
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger


class TransformerTrainingPipeline:

    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            config = ConfigurationManager()

            trainer_config = config.get_transformer_trainer_config()

            trainer = TransformerTrainer(trainer_config)

            trainer.initiate_transformer_training()

            logger.info("Transformer Training Pipeline completed successfully.")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TransformerTrainingPipeline()
    pipeline.run_pipeline()