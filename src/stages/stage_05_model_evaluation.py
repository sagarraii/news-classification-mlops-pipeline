import sys

from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException


STAGE_NAME = "MODEL EVALUATION"


class ModelEvaluationTrainingPipeline:

    def __init__(self):
        self.config = ConfigurationManager()

    def main(self):
        logger.info("Initializing ModelEvaluation component.")

        evaluation = ModelEvaluation(
            transformation_config=self.config.get_data_transformation_config(),
            model_trainer_config=self.config.get_model_trainer_config(),
            model_eval_config=self.config.get_model_evaluation_config(),
        )

        logger.info("Starting model evaluation.")

        report_path = evaluation.initiate_model_evaluation()

        logger.info("Model evaluation completed successfully.")
        logger.info("Evaluation report saved at: %s", report_path)


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("Stage Started : %s", STAGE_NAME)
        logger.info("=" * 60)

        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.main()

        logger.info("=" * 60)
        logger.info("Stage Completed : %s", STAGE_NAME)
        logger.info("=" * 60)

    except Exception as e:
        logger.exception("Model evaluation stage failed.")
        raise CustomException(e, sys)