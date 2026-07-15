import sys

from src.components.data_validation import DataValidation
from src.config.configuration import ConfigurationManager

from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger

STAGE_NAME = "DATA VALIDATION"


class DataValidationTrainingPipeline:

    def __init__(self):
        config = ConfigurationManager()
        self.data_validation_config = config.get_data_validation_config()


    def main(self):
        data_validation = DataValidation(self.data_validation_config)
        data_validation.initiate_data_validation()


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info(f"Stage Started : {STAGE_NAME}")
        logger.info("=" * 60)

        pipeline = DataValidationTrainingPipeline()
        pipeline.main()

        logger.info("=" * 60)
        logger.info(f"Stage Completed : {STAGE_NAME}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(str(e))
        raise CustomException(e, sys)