import re
import sys

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import TextPreprocessorConfig


class TextPreprocessor:
    
    def __init__(self, config: TextPreprocessorConfig) -> None:
      self.config = config
      try: 
        self.stop_words = (
            set(stopwords.words("english"))
            if self.config.remove_stopwords
            else set()
        )

        self.lemmatizer = (
            WordNetLemmatizer()
            if self.config.lemmatize
            else None
        )

      except LookupError as e:
        raise CustomException(
            Exception(
                "Required NLTK resources are missing. "
                "Run download_nltk_resources() before using TextPreprocessor."
            ),
            sys,
        )

      logger.info("TextPreprocessor initialized.")


    def clean_text(self, text: str) -> str:
      
      try:
        text = "" if text is None else str(text)
        text = text.lower()

        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
      
      except Exception as e:
         raise CustomException(e, sys)
      

    def tokenize(self, text: str) -> list[str]:
        try:
            tokens = word_tokenize(text)
            return [token for token in tokens if len(token) >= self.config.min_token_length]
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        try:
            return [token for token in tokens if token not in self.stop_words]
        
        except Exception as e:
            raise CustomException(e, sys)
    

    def lemmatize(self, tokens: list[str]) -> list[str]:
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        
        except Exception as e:
            raise CustomException(e, sys)


    def preprocess(self, text: str) -> tuple[str, list[str]]:
        try:
            text = self.clean_text(text)
            tokens = self.tokenize(text)

            if self.config.remove_stopwords:
                tokens = self.remove_stopwords(tokens)

            if self.config.lemmatize:
                tokens = self.lemmatize(tokens)

            cleaned_text = " ".join(tokens)

            return cleaned_text, tokens

        except Exception as e:
            logger.exception("Error during text preprocessing.")
            raise CustomException(e, sys)

    
    def preprocess_series(self, series : pd.Series) -> tuple[pd.Series, list[list[str]]]:
        try:
            results = series.apply(self.preprocess)

            cleaned_series = results.apply(lambda x: x[0])
            token_lists = results.apply(lambda x: x[1]).tolist()

            logger.info(f"Successfully preprocessed {len(series)} documents.")

            return cleaned_series, token_lists

        except Exception as e:
            logger.exception("Error while preprocessing text series.")
            raise CustomException(e, sys)

'''
if __name__ == "__main__":

  from src.config.configuration import ConfigurationManager

  config = ConfigurationManager().get_text_preprocessor_config()

  preprocessor = TextPreprocessor(config)

  sample = "The Prime Minister announced new economic policies today at Parliament."

  cleaned, tokens = preprocessor.preprocess(sample)

  print(cleaned)
  print(tokens)

'''