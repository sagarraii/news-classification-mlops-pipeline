from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from typing import Dict

def calculate_metrics(preds, labels) -> Dict[str, float]:
    
    return {
    "accuracy": accuracy_score(labels, preds),
    "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
    "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0)
    }
