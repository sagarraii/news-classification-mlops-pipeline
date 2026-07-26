import torch
from tqdm import tqdm


class Trainer:

    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, device,):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        predictions = []
        labels = []

        with torch.no_grad():
            progress_bar = tqdm(self.valid_loader, desc="Validation", leave=False)

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)

                loss = outputs.loss
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits,dim=1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / len(self.valid_loader)

        return avg_loss, predictions, labels