import sys

from datasets import load_dataset

from src.logger.custom_logger import logger
from src.exception.custom_exception import CustomException
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_dataset(self):

        try:

            download_path = self.config.dataset_download_dir
            download_path.mkdir(parents=True, exist_ok=True)

            train_file = download_path / "train.csv"
            test_file = download_path / "test.csv"

            if train_file.exists() and test_file.exists():
                logger.info("Dataset already exists. Skipping download.")
                return

            logger.info("Downloading dataset...")

            dataset = load_dataset(self.config.dataset_name)

            train_dataset = dataset[self.config.train_dataset]
            test_dataset = dataset[self.config.test_dataset]

            train_dataset.to_pandas().to_csv(train_file, index=False, encoding="utf-8")
            test_dataset.to_pandas().to_csv(test_file, index=False, encoding="utf-8")

            logger.info("Train and test datasets saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)