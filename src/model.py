import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MultiTaskDistilBERT(nn.Module):
    """
    Shared DistilBERT encoder + separate classification heads
    for Sentiment, Request Type, Product Area, Priority, Satisfaction.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: dict = None,          # e.g. {"sentiment": 3, "request_type": 6, ...}
        dropout: float = 0.1,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # One classification head per task
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, n_labels)
            for task, n_labels in num_labels.items()
        })

        self.task_names = list(num_labels.keys())

    def forward(self, input_ids, attention_mask, **kwargs):
        # Shared encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_embedding = self.dropout(cls_embedding)

        # Separate logits for each task
        logits = {
            task: self.heads[task](cls_embedding)
            for task in self.task_names
        }

        return logits


def get_num_labels(encoders: dict) -> dict:
    """Helper to get number of classes from the label encoders."""
    return {field: len(le.classes_) for field, le in encoders.items()}