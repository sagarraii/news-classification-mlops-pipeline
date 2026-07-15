import sys

from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation

from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger

STAGE_NAME = "DATA TRANSFORMATION"


class DataTransformationTrainingPipeline:

    def __init__(self):
        self.config = ConfigurationManager()


    def main(self):
        logger.info("Initializing DataTransformation component.")

        transformation = DataTransformation(
            tp_config=self.config.get_text_preprocessor_config(),
            fb_config=self.config.get_feature_builder_config(),
            dt_config=self.config.get_data_transformation_config(),
        )

        logger.info("Starting data transformation pipeline.")

        transformation.initiate_data_transformation()

        logger.info("Data transformation pipeline completed successfully.")


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info(f"Stage Started : {STAGE_NAME}")
        logger.info("=" * 60)

        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()

        logger.info("=" * 60)
        logger.info(f"Stage Completed : {STAGE_NAME}")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception("Data transformation stage failed.")
        raise CustomException(e, sys)