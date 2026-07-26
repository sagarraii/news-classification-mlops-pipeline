import sys
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException

from src.utils.common import save_bin
from src.entity.config_entity import FeatureBuilderConfig


class FeatureBuilder:
    def __init__(self, config: FeatureBuilderConfig) -> None:
        self.config = config


    def build_tfidf(self, X_train_text: pd.Series, X_validation_text: pd.Series, X_test_text: pd.Series) -> tuple[csr_matrix, csr_matrix, csr_matrix]:

        try: 
            self.vectorizer = TfidfVectorizer(
                    max_features=self.config.tfidf.max_features,
                    ngram_range=self.config.tfidf.ngram_range,
                    min_df=self.config.tfidf.min_df,
                    max_df=self.config.tfidf.max_df,
                )

            X_train_tfidf = self.vectorizer.fit_transform(X_train_text)
            X_validation_tfidf = self.vectorizer.transform(X_validation_text)
            X_test_tfidf  = self.vectorizer.transform(X_test_text)

            save_bin(self.vectorizer, self.config.vectorizer_path)

            logger.info(f"TF-IDF vocab size: {len(self.vectorizer.vocabulary_)}")

            logger.info(f"Train TF-IDF shape: {X_train_tfidf.shape}")
            logger.info(f"Validation TF-IDF shape: {X_validation_tfidf.shape}")
            logger.info(f"Test TF-IDF shape: {X_test_tfidf.shape}")

            return (X_train_tfidf, X_validation_tfidf ,X_test_tfidf)
        
        except Exception as e:
            logger.exception("Error while building TF-IDF features.")
            raise CustomException(e, sys)
    

    