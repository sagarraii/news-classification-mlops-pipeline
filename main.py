from src.stages.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.logger.custom_logger import logger

STAGE_NAME = "DATA INGESTION"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")

    obj = DataIngestionTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e