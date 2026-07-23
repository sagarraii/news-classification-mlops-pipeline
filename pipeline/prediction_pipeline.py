import sys
import pandas as pd

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException
from src.utils.common import load_object, load_bin

from src.components.text_preprocessor import TextPreprocessor

from src.entity.config_entity import (
    TextPreprocessorConfig,
    FeatureBuilderConfig,
    TraditionalModelTrainerConfig,
)


class PredictionPipeline:
    def __init__(
        self,
        tp_config: TextPreprocessorConfig,
        fb_config: FeatureBuilderConfig,
        tmt_config: TraditionalModelTrainerConfig,
    ) -> None:
        try:
            logger.info("Initializing Prediction Pipeline...")

            # Initialize text preprocessor
            self.preprocessor = TextPreprocessor(tp_config)

            # Load saved artifacts
            self.vectorizer = load_bin(fb_config.vectorizer_path)
            self.model = load_object(tmt_config.best_model_path)

            self.label_map = {
                1: "World",
                2: "Sports",
                3: "Business",
                4: "Sci/Tech",
            }

            logger.info("Prediction Pipeline initialized successfully.")

        except Exception as e:
            logger.exception("Failed to initialize Prediction Pipeline.")
            raise CustomException(e, sys)

    def predict(self, text: str) -> str:
        try:
            # Preprocess text
            cleaned_text, _ = self.preprocessor.preprocess(text)

            # Vectorize
            features = self.vectorizer.transform([cleaned_text])

            # Predict
            prediction = self.model.predict(features)[0]

            return self.label_map.get(
                int(prediction),
                f"Unknown ({prediction})"
            )

        except Exception as e:
            logger.exception("Prediction failed.")
            raise CustomException(e, sys)