# ingestion of ml flow is still pending and i will do it later

import sys
import pandas as pd

from typing import Any
from pathlib import Path
from dataclasses import asdict

from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import StratifiedKFold

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

from src.utils.common import (load_sparse_matrix, 
                              load_numpy_array_data,
                              save_object, 
                              write_yaml
                              )

from src.entity.config_entity import(
    DataTransformationConfig,
    TrainingData,
    TraditionalModelTrainerConfig,
    ModelTrainerParams
)


class TraditionalModelTrainer:
    def __init__(self, transformation_config: DataTransformationConfig, model_trainer_config: TraditionalModelTrainerConfig, train_params_config: ModelTrainerParams):
        self.transformation_config = transformation_config
        self.model_trainer_config = model_trainer_config
        self.train_params_config = train_params_config


    def load_training_data(self) -> TrainingData:
        try:
            
            logger.info("Starting model training: loading training and validation data.")

            X_train = load_sparse_matrix(self.transformation_config.processed_train_data_path)
            y_train = load_numpy_array_data(self.transformation_config.processed_train_target_path)
            X_valid = load_sparse_matrix(self.transformation_config.processed_validation_data_path)
            y_valid = load_numpy_array_data(self.transformation_config.processed_validation_target_path)
            
            logger.info(
                f"Loaded X_train: {X_train.shape}, "
                f"X_valid: {X_valid.shape}, "
                f"y_train: {y_train.shape}, "
                f"y_valid: {y_valid.shape}"
            )

            return TrainingData( X_train=X_train, 
                                X_valid=X_valid, 
                                y_train=y_train, 
                                y_valid=y_valid) # new to try approach
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initialize_models(self) -> dict[str, BaseEstimator]:
        try:
            logger.info("Initializing candidate machine learning models.")

            random_state=self.train_params_config.random_search.random_state
            models = {
                "Logistic Regression": LogisticRegression(random_state=random_state),
                "Multinomial Naive Bayes": MultinomialNB(),
                "LinearSVC": LinearSVC(random_state=random_state),
                "LightGBM": LGBMClassifier(random_state=random_state)

            }

            logger.info(f"Initialized {len(models)} models for hyperparameter tuning.")

            return models 
        
        except Exception as e:
            raise CustomException (e, sys)


    def get_param_grids(self) -> dict:
        try:

            logger.info("Loading hyperparameter search spaces for all models.")

            return {
                "Logistic Regression": asdict(self.train_params_config.logistic_regression),
                "Multinomial Naive Bayes": asdict(self.train_params_config.multinomial_naive_bayes),
                "LinearSVC": asdict(self.train_params_config.linear_svm),
                "LightGBM": asdict(self.train_params_config.lightgbm),
            }
        
        except Exception as e:
            raise CustomException (e, sys)


    def tune_model(self, model_name, model, params_grid, X_train, y_train):
        try:
            logger.info(f"Starting RandomizedSearchCV for {model_name}")

            cv = StratifiedKFold(
                n_splits=self.train_params_config.random_search.cv,
                shuffle=True,
                random_state=self.train_params_config.random_search.random_state,
            )

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params_grid,
                n_iter=self.train_params_config.random_search.n_iter,
                return_train_score=True,
                cv = cv,
                scoring=self.train_params_config.random_search.scoring,
                random_state=self.train_params_config.random_search.random_state,
                n_jobs=self.train_params_config.random_search.n_jobs,
                verbose=self.train_params_config.random_search.verbose,
            )
            logger.info(f"Performing hyperparameter tuning for {model_name}.")

            random_search.fit(X_train, y_train)
            logger.info(
                f"Hyperparameter tuning completed for {model_name}. "
                f"Best CV MACRO_F1 Score: {random_search.best_score_:.4f}"
            )

            return random_search

        except Exception as e:
            raise CustomException(e, sys)


    def evaluate_model(self, model, X_valid, y_valid) -> dict:
        try:
            logger.info("Evaluating model on validation dataset.")

            y_pred = model.predict(X_valid)

            accuracy = accuracy_score(y_valid, y_pred)
            macro_precision = precision_score(y_valid, y_pred, average="macro", zero_division=0)
            macro_recall = recall_score(y_valid, y_pred, average="macro", zero_division=0)
            macro_f1 = f1_score(y_valid, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_valid, y_pred, average="weighted", zero_division=0)
            
            logger.info(
                f"Validation Metrics - "
                f"Accuracy: {accuracy:.4f}, "
                f"Macro Precision: {macro_precision:.4f}, "
                f"Macro Recall: {macro_recall:.4f}, "
                f"Macro F1: {macro_f1:.4f}, "
                f"Weighted F1: {weighted_f1:.4f}, "

            )

            return {
                "Validation Accuracy": round(accuracy, 4),
                "Validation Macro Precision": round(macro_precision, 4),
                "Validation Macro Recall": round(macro_recall, 4),
                "Validation Macro F1": round(macro_f1, 4),
                "Validation Weighted F1": round(weighted_f1, 4),

                }

        except Exception as e:
            raise CustomException(e, sys)


    def save_search_results(self, cv_results, model_name):
        try:
            df = pd.DataFrame(cv_results)

            safe_name = (model_name.lower().replace(" ", "_"))
            file_path = (self.model_trainer_config.search_results_dir/ f"{safe_name}_search_results.csv")
            df.to_csv(file_path, index=False)

            logger.info(f"Randomized search results saved to: {file_path}")

        except Exception as e:
            raise CustomException(e, sys)
        
    def save_artifacts(self, best_model, best_model_name, best_validation_macro_f1, best_params, report):
        try:
            save_object(file_path=self.model_trainer_config.best_model_path, obj=best_model,)
            logger.info(f"Best model saved at: {self.model_trainer_config.best_model_path}")

            # write yaml is in "W" mode which over rite the existing file, so we dont need replace argument here in write yaml
            
            write_yaml(file_path=self.model_trainer_config.model_report_path, content=report,)
            logger.info(f"Model comparison report saved at: {self.model_trainer_config.model_report_path}")

            write_yaml(file_path=self.model_trainer_config.best_params_path,content={
                    "model_name": best_model_name,
                    "validation_macro_f1": float(best_validation_macro_f1),
                    "best_params": best_params,
                },
            )

            logger.info(f"Best model parameters saved at: {self.model_trainer_config.best_params_path}")

        except Exception as e:
            raise CustomException (e, sys)


    def initiate_model_trainer(self) -> Path:
        try:
            logger.info("Starting model training pipeline.")

            data = self.load_training_data()

            X_train = data.X_train
            X_valid = data.X_valid
            y_train = data.y_train
            y_valid = data.y_valid

            models = self.initialize_models()
            param_grids = self.get_param_grids()
            logger.info("Beginning model selection and hyperparameter tuning.")

            results: list[dict[str, Any]] = []

            best_model = None
            best_model_name = None
            best_validation_macro_f1 = float("-inf")
            best_params = None

            for model_name, model in models.items():

                logger.info(f"Training {model_name}")
                

                search = self.tune_model(
                    model_name=model_name,
                    model=model,
                    params_grid=param_grids[model_name],
                    X_train=X_train,
                    y_train=y_train,
                )

                logger.info(
                    f"Best parameters found for {model_name}: "
                    f"{search.best_params_}"
                )

                self.save_search_results(search.cv_results_, model_name,)
                metrics = self.evaluate_model(search.best_estimator_, X_valid, y_valid,)

                logger.info(
                    f"{model_name} achieved Validation Macro F1 Score: "
                    f"{metrics['Validation Macro F1']:.4f}"
                )

                results.append(
                    {"Model": model_name, "Best CV Macro F1": round(search.best_score_, 4,),
                    **metrics, "Best Params": search.best_params_,
                    }
                )

                if (metrics["Validation Macro F1"] > best_validation_macro_f1):
                    best_validation_macro_f1 = (metrics["Validation Macro F1"])
                    best_model = search.best_estimator_
                    best_model_name = model_name
                    best_params = search.best_params_
                    logger.info(
                        f"{model_name} is the current best model "
                        f"with Validation Macro F1 Score: {best_validation_macro_f1:.4f}"
                    )

            report_df = (pd.DataFrame(results).sort_values(by="Validation Macro F1", ascending=False,).reset_index(drop=True))
            
            logger.info("Model comparison report generated successfully.")

            self.save_artifacts(
                best_model=best_model,
                best_model_name=best_model_name,
                best_validation_macro_f1=best_validation_macro_f1,
                best_params=best_params,
                report=report_df.to_dict(
                    orient="records"
                ),
            )

            
            logger.info(
                f"Model training completed successfully. "
                f"Selected best model: {best_model_name} "
                f"(Validation Macro F1: {best_validation_macro_f1:.4f})."
            )

            return self.model_trainer_config.best_model_path

        except Exception as e:
            raise CustomException(e, sys)
        
