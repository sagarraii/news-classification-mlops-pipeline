import sys
import pandas as pd
import numpy as np

from pathlib import Path

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from src.utils.nltk_utils import download_nltk_resources
from src.utils.common import (save_sparse_matrix, save_numpy_array_data)

from src.components.text_preprocessor import TextPreprocessor
from src.components.feature_builder import FeatureBuilder

from src.entity.config_entity import (TextPreprocessorConfig, 
                                      FeatureBuilderConfig ,
                                      DataTransformationConfig)

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

# for temperory test
# from src.config.configuration import ConfigurationManager

class DataTransformation:
    def __init__(self, 
                 tp_config: TextPreprocessorConfig, 
                 fb_config: FeatureBuilderConfig ,
                 dt_config:DataTransformationConfig) -> None:
        
        self.preprocessor = TextPreprocessor(tp_config)
        self.feature_builder = FeatureBuilder(fb_config)
        self.data_transform_config = dt_config
    

    @staticmethod
    def read_data(train_path: Path) -> pd.DataFrame:
        try:
            logger.info(f"Loading dataset from '{train_path}'.")

            df = pd.read_csv(train_path)

            logger.info(f"Dataset loaded successfully with {len(df)} records.")          
            return df
        
        except Exception as e:
            raise CustomException (e, sys)
        
    
    def combine_text(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            text_columns = self.data_transform_config.text_columns
            dataframe["text"] = (dataframe[text_columns[0]].fillna("").astype(str) + " " + dataframe[text_columns[1]].fillna("").astype(str))

            logger.info("Combined title and description into a single 'text' column.")
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def drop_text_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            columns_to_drop = self.data_transform_config.text_columns

            dataframe = dataframe.drop(columns=columns_to_drop, errors="ignore")
            assert "text" in dataframe.columns

            logger.info(f"Dropped columns: {columns_to_drop}")
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def split_features(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        try:
            target_col = self.data_transform_config.target_column

            X = dataframe.drop(columns=target_col)
            y = dataframe[target_col]

            return X, y
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def train_test_split_data(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        try:
            X, y = self.split_features(dataframe)

            test_size = self.data_transform_config.test_size
            validation_size = self.data_transform_config.validation_size
            random_state = self.data_transform_config.random_state

            X_train, X_temp, y_train, y_temp, = train_test_split(X,y, test_size=test_size, random_state=random_state, stratify=y)

            X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=random_state, stratify=y_temp)

            logger.info(
                f"Feature shapes (X)- "
                f"Train: {X_train.shape}, "
                f"Validation: {X_valid.shape}, "
                f"Test: {X_test.shape}"
            )

            logger.info(
                f"Target shapes (y) - "
                f"Train: {y_train.shape}, "
                f"Validation: {y_valid.shape}, "
                f"Test: {y_test.shape}"
            )

            return X_train, X_valid, X_test, y_train, y_valid, y_test
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def preprocess_data(self, X_train: pd.DataFrame, X_validation: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        try:
            X_train_clean, _ = self.preprocessor.preprocess_series(X_train["text"])
            X_validation_clean, _ = self.preprocessor.preprocess_series(X_validation["text"])
            X_test_clean, _ = self.preprocessor.preprocess_series(X_test["text"])

            return (X_train_clean, X_validation_clean, X_test_clean)
        
        except Exception as e:
            raise CustomException (e, sys)


    def build_features(self, X_train_clean: pd.Series, X_validation_clean: pd.Series, X_test_clean: pd.Series ) -> tuple[csr_matrix, csr_matrix, csr_matrix,]:
        try:
            X_train_tfidf, X_validation_tfidf, X_test_tfidf = self.feature_builder.build_tfidf( X_train_clean, X_validation_clean, X_test_clean,)

            return (X_train_tfidf, X_validation_tfidf, X_test_tfidf)
        
        except Exception as e:
            raise CustomException(e, sys)


    def save_transformed_data(self, 
                            X_train: csr_matrix, 
                            y_train: pd.Series, 
                            X_validation: csr_matrix, 
                            y_valid: pd.Series,  
                            X_test: csr_matrix, 
                            y_test: pd.Series) -> None:
        try:

            save_sparse_matrix(self.data_transform_config.processed_train_data_path, X_train,)
            save_sparse_matrix(self.data_transform_config.processed_validation_data_path, X_validation,)
            save_sparse_matrix(self.data_transform_config.processed_test_data_path, X_test,)

            save_numpy_array_data(self.data_transform_config.processed_train_target_path, np.asarray(y_train),)
            save_numpy_array_data(self.data_transform_config.processed_validation_target_path, np.asarray(y_valid),)
            save_numpy_array_data(self.data_transform_config.processed_test_target_path, np.asarray(y_test),)

            logger.info("Processed features and target arrays saved successfully.")
            
        except Exception as e:
                raise CustomException(e, sys)


    def initiate_data_transformation(self) -> None:
        try:
            logger.info("=" * 50)
            logger.info("Data Transformation Started")
            logger.info("=" * 50)

            download_nltk_resources()

            dataframe = self.read_data(self.data_transform_config.train_path)
            dataframe = self.combine_text(dataframe)
            dataframe = self.drop_text_columns(dataframe)

            (X_train, X_validation, X_test, y_train, y_validation, y_test,) = self.train_test_split_data(dataframe)
            (X_train_clean, X_validation_clean, X_test_clean,) = self.preprocess_data(X_train, X_validation, X_test,)
            (X_train_tfidf, X_validation_tfidf, X_test_tfidf,) = self.build_features(X_train_clean, X_validation_clean, X_test_clean,)

            self.save_transformed_data(X_train_tfidf, y_train, X_validation_tfidf, y_validation, X_test_tfidf, y_test,)

            logger.info("=" * 50)
            logger.info("Data Transformation Completed Successfully")
            logger.info("=" * 50)

        except Exception as e:
            logger.exception("Error occurred during data transformation.")
            raise CustomException(e, sys)
        

