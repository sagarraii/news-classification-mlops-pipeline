# Note: when i test my model with test.csv which is totally raw and uncouched in that case i'll need 
# text_preprocessor and feature_builder.py for transformaing my raw data in vectors
# and for that we need to create another defination named transformation so we can 
# initialize the required extracted file and transformed our raw data here itself only.

import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from pathlib import Path

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

from src.entity.config_entity import (
    DataTransformationConfig,
    TraditionalModelTrainerConfig,
    TestingData,
    ModelEvaluationConfig)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.utils.common import(
    load_numpy_array_data,
    load_sparse_matrix,
    load_object,
    write_yaml
)

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

CLASS_IDS = list(LABEL_MAP.keys())
CLASS_NAMES = list(LABEL_MAP.values())

class ModelEvaluation:
    def __init__(self, transformation_config: DataTransformationConfig, model_trainer_config: TraditionalModelTrainerConfig, model_eval_config: ModelEvaluationConfig) -> None:
        self.transformation_config = transformation_config
        self.model_trainer_config = model_trainer_config
        self.model_eval_config = model_eval_config


    def read_data(self) -> TestingData:
        try:
            logger.info("Loading test dataset.")

            X_test = load_sparse_matrix(self.transformation_config.processed_test_data_path) 
            y_test = load_numpy_array_data(self.transformation_config.processed_test_target_path)

            logger.info(f"Loaded X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            return TestingData(
                X_test=X_test,
                y_test=y_test
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def load_model(self) -> BaseEstimator:
        try:
            logger.info(f"Loading trained model from {self.model_trainer_config.best_model_path}")

            best_model = load_object(self.model_trainer_config.best_model_path)

            logger.info("Best model loaded successfully.")
            return best_model
                    
        except Exception as e:
            raise CustomException(e, sys)

 
    def evaluate(self, model, X_test, y_test) -> tuple[dict, np.ndarray]:
        try:
            logger.info("Evaluating model on Test Dataset.")

            y_pred = model.predict(X_test)

            metrics = {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "Macro_F1": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
                "Weighted_F1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            }

            logger.info(
                f"Test Metrics - "
                f"Accuracy: {metrics['Accuracy']:.4f}, "
                f"Macro Precision: {metrics['Precision']:.4f}, "
                f"Macro Recall: {metrics['Recall']:.4f}, "
                f"Macro F1: {metrics['Macro_F1']:.4f}, "
                f"Weighted F1: {metrics['Weighted_F1']:.4f} "

            )

            return metrics, y_pred

        except Exception as e:
            raise CustomException(e, sys)


    def generate_classification_report(self, y_test, y_pred) -> dict:
        try:

            cr = classification_report(y_test, y_pred, labels=CLASS_IDS, target_names=CLASS_NAMES, output_dict=True, zero_division=0,)

            logger.info("Classification report generated successfully.")
            return cr

        except Exception as e:
            raise CustomException(e, sys)


    def save_report(self, report, metrics) -> None:
        try:

            write_yaml(file_path=self.model_eval_config.classification_report_path, content=report,)

            logger.info(
                f"Classification report saved to "
                f"{self.model_eval_config.classification_report_path}"
            )

            write_yaml(file_path=self.model_eval_config.evaluation_report_path, content=metrics,)

            logger.info(
                f"Evaluation Report saved to "
                f"{self.model_eval_config.evaluation_report_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)


    def save_model_acceptance(self, metrics) -> None:
        try:
            threshold = 0.65
            accepted = metrics["Macro_F1"] >= threshold

            report = {
                "accepted": accepted,
                "criterion": f"Macro_F1 >= {threshold}",
                "actual_f1": metrics["Macro_F1"],
            }

            write_yaml(file_path=self.model_eval_config.evaluation_summary_path, content=report,)

            logger.info(
                f"Model acceptance report saved to "
                f"{self.model_eval_config.evaluation_summary_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)
        

    def save_confusion_matrix(self, y_test, y_pred) -> None:
        try:

            cm = confusion_matrix(y_test, y_pred, labels=CLASS_IDS)

            plt.figure(figsize=(6, 6))
            plt.imshow(cm, interpolation="nearest", aspect="equal")
            plt.title("Confusion Matrix")
            plt.colorbar()

            plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
            plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

            plt.xlabel("Predicted")
            plt.ylabel("Actual")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")

            plt.tight_layout()
            self.model_eval_config.confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True,)
            plt.savefig(self.model_eval_config.confusion_matrix_path, bbox_inches="tight",)
            plt.close()
            logger.info(
                f"Confusion matrix saved to "
                f"{self.model_eval_config.confusion_matrix_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)
     
        
    def initiate_model_evaluation(self) -> Path:
        try:
            logger.info("Starting model evaluation pipeline.")

            data = self.read_data()
            X_test = data.X_test
            y_test = data.y_test

            model = self.load_model()

            metrics, y_pred = self.evaluate(model=model, X_test=X_test, y_test=y_test,)

            report = self.generate_classification_report(y_test=y_test, y_pred=y_pred,)
            self.save_report(report=report, metrics=metrics,)
            self.save_model_acceptance(metrics)
            self.save_confusion_matrix(y_test=y_test, y_pred=y_pred,)

            logger.info("Model evaluation completed successfully.")
            return self.model_eval_config.evaluation_report_path
        
        except Exception as e:
            raise CustomException(e, sys)