import os
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.entity.config_entity import TransformerTrainerConfig
from src.exception.custom_exception import CustomException
from src.logger.custom_logger import logger
from src.utils.dataset import NewsDataset
from src.utils.metrics import calculate_metrics
from src.utils.trainer import Trainer


class TransformerTrainer:

    def __init__(self, config: TransformerTrainerConfig) -> None:
        self.config = config

    def read_data(self, train_data_path: Path) -> pd.DataFrame:
        try:
            logger.info(f"Loading dataset from '{train_data_path}'.")
            dataframe = pd.read_csv(train_data_path)
            dataframe["text"] = (dataframe["title"].fillna("") + " " + dataframe["description"].fillna(""))
            dataframe.drop(columns=["title", "description"], inplace=True)
            logger.info(f"Dataset loaded successfully with {len(dataframe)} records.")
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)

    def split_data(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_df, valid_df = train_test_split(
                dataframe,
                test_size=self.config.validation_size,
                random_state=self.config.random_state,
                stratify=dataframe["label"])
            
            logger.info("Train Validation split completed.")
            return train_df, valid_df
        except Exception as e:
            raise CustomException(e, sys)

    def save_checkpoint(self, epoch, model, optimizer, scheduler, best_f1, metrics):
        try:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "metrics": metrics}

            torch.save(checkpoint, os.path.join(self.config.model_dir, "best_checkpoint.pt"))
            logger.info("Checkpoint saved successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def train(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        try:
            torch.manual_seed(self.config.random_state)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.random_state)

            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # ---- remove
            logger.info(f"Unique labels: {sorted(train_df['label'].unique())}")
            logger.info(f"Number of labels: {train_df['label'].nunique()}")
            # ---- remove upper part
            train_df["label"] = train_df["label"] - 1
            valid_df["label"] = valid_df["label"] - 1
            logger.info(f"Using device : {device}")
            model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name, num_labels=4)
            model.to(device)
            os.makedirs(self.config.model_dir, exist_ok=True)
            os.makedirs(self.config.tokenizer_dir, exist_ok=True)

            train_dataset = NewsDataset(texts=train_df["text"].tolist(), labels=train_df["label"].tolist(), tokenizer=tokenizer, max_length=self.config.max_length)
            valid_dataset = NewsDataset(texts=valid_df["text"].tolist(), labels=valid_df["label"].tolist(),tokenizer=tokenizer, max_length=self.config.max_length)
            pin_memory = torch.cuda.is_available()

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=pin_memory, num_workers=self.config.num_workers)
            valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size, shuffle=False, pin_memory=pin_memory, num_workers=self.config.num_workers)

            #once solved i ll remmove this : solved
            logger.info(f"learning_rate = {self.config.learning_rate} ({type(self.config.learning_rate)})")
            logger.info(f"weight_decay = {self.config.weight_decay} ({type(self.config.weight_decay)})")
            #-----------------remove upper part

            optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            total_training_steps = (len(train_loader) * self.config.num_epochs)

            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(0.1 * total_training_steps), num_training_steps=total_training_steps)
            trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, valid_loader=valid_loader, device=device,)

            best_f1 = 0.0
            counter = 0

            for epoch in range(self.config.num_epochs):
                logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}]")

                train_loss = trainer.train_epoch()
                valid_loss, preds, labels = trainer.validate()
                metrics = calculate_metrics(preds, labels)
                current_f1 = metrics["macro_f1"]

                logger.info(
                    f"Train Loss : {train_loss:.4f} | "
                    f"Validation Loss : {valid_loss:.4f} | "
                    f"Accuracy : {metrics['accuracy']:.4f} | "
                    f"Precision : {metrics['macro_precision']:.4f} | "
                    f"Recall : {metrics['macro_recall']:.4f} | "
                    f"Validation F1 : {current_f1:.4f} |"
                    f"Weighted F1 : {metrics['weighted_f1']:.4f}")

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    counter = 0
                    os.makedirs(self.config.model_dir,exist_ok=True)
                    os.makedirs(self.config.tokenizer_dir, exist_ok=True)             
                    model.save_pretrained(self.config.model_dir)
                    tokenizer.save_pretrained(self.config.tokenizer_dir)
                    self.save_checkpoint(epoch, model, optimizer, scheduler, best_f1, metrics)
                    logger.info("Best model saved.")
                else:
                    counter += 1

                if counter >= self.config.patience:
                    logger.info("Early stopping triggered.")
                    break
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformer_training(self):
        try:
            dataframe = self.read_data(self.config.train_data_path)
            train_df, valid_df = self.split_data(dataframe)
            self.train(train_df, valid_df)
            logger.info("Transformer training completed successfully.")
        except Exception as e:
            raise CustomException(e, sys)