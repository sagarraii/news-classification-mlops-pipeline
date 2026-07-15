import sys

from src.config.configuration import ConfigurationManager
from src.components.traditional_model_trainer import TraditionalModelTrainer

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException


STAGE_NAME = "MODEL TRAINING"


class ModelTrainingPipeline:

    def __init__(self):
        self.config = ConfigurationManager()

    def main(self):
        logger.info("Initializing TraditionalModelTrainer component.")

        trainer = TraditionalModelTrainer(
            transformation_config=self.config.get_data_transformation_config(),
            model_trainer_config=self.config.get_model_trainer_config(),
            train_params_config=self.config.get_model_trainer_params(),
        )

        logger.info("Starting model training.")

        best_model_path = trainer.initiate_model_trainer()

        logger.info("Model training completed successfully.")
        logger.info("Best model saved at: %s", best_model_path)


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Stage Started : %s", STAGE_NAME)
        logger.info("=" * 60)

        pipeline = ModelTrainingPipeline()
        pipeline.main()

        logger.info("=" * 60)
        logger.info("Stage Completed : %s", STAGE_NAME)
        logger.info("=" * 60)

    except Exception as e:
        logger.exception("Model training stage failed.")
        raise CustomException(e, sys)