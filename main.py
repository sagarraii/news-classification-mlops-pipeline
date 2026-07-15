import sys

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

from src.stages.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.stages.stage_02_data_validation import DataValidationTrainingPipeline
from src.stages.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.stages.stage_04_model_training import ModelTrainingPipeline
from src.stages.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


STAGES = [
    ("DATA INGESTION", DataIngestionTrainingPipeline),
    ("DATA VALIDATION", DataValidationTrainingPipeline),
    ("DATA TRANSFORMATION", DataTransformationTrainingPipeline),
    ("MODEL TRAINING", ModelTrainingPipeline),
    ("MODEL EVALUATION", ModelEvaluationTrainingPipeline),
]


def run_stage(stage_name: str, pipeline_class):
    logger.info("=" * 60)
    logger.info("Stage Started : %s", stage_name)
    logger.info("=" * 60)

    pipeline = pipeline_class()
    pipeline.main()

    logger.info("=" * 60)
    logger.info("Stage Completed : %s", stage_name)
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        logger.info("#" * 70)
        logger.info("NLP NEWS CLASSIFICATION TRAINING PIPELINE STARTED")
        logger.info("#" * 70)

        for stage_name, pipeline_class in STAGES:
            run_stage(stage_name, pipeline_class)

        logger.info("#" * 70)
        logger.info("NLP NEWS CLASSIFICATION TRAINING PIPELINE COMPLETED")
        logger.info("#" * 70)

    except Exception as e:
        logger.exception("Training pipeline failed.")
        raise CustomException(e, sys)