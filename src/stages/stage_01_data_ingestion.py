import sys

from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion

from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger

STAGE_NAME = "DATA INGESTION"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_dataset()


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info(f"Stage Started : {STAGE_NAME}")
        logger.info("=" * 60)

        obj = DataIngestionTrainingPipeline()
        obj.main()

        logger.info("=" * 60)
        logger.info(f"Stage Completed : {STAGE_NAME}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(str(e))
        raise CustomException(e, sys)