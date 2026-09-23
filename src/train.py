import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import numpy as np
import joblib
from pathlib import Path
import json

from dataset import prepare_dataset, LABEL_FIELDS
from model import MultiTaskDistilBERT, get_num_labels

# ====================== CONFIG ======================
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("models/multitask_distilbert")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")


def compute_metrics(all_preds: dict, all_labels: dict):
    """Calculate field-wise accuracy, F1 and Exact Match."""
    metrics = {}
    exact_match_count = 0
    total = len(next(iter(all_labels.values())))

    for field in LABEL_FIELDS:
        preds = all_preds[field]
        labels = all_labels[field]

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")

        metrics[f"{field}_accuracy"] = round(acc, 4)
        metrics[f"{field}_f1"] = round(f1, 4)

    # Exact Match = all 5 fields correct for a ticket
    for i in range(total):
        if all(all_preds[field][i] == all_labels[field][i] for field in LABEL_FIELDS):
            exact_match_count += 1

    metrics["exact_match"] = round(exact_match_count / total, 4)
    return metrics


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = {field: [] for field in LABEL_FIELDS}
    all_labels = {field: [] for field in LABEL_FIELDS}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            for field in LABEL_FIELDS:
                preds = torch.argmax(logits[field], dim=-1).cpu().numpy()
                labels = batch[f"{field}_label"].cpu().numpy()

                all_preds[field].extend(preds)
                all_labels[field].extend(labels)

    return compute_metrics(all_preds, all_labels)


def train():
    # ----- Load data -----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset, encoders = prepare_dataset(tokenizer)
    num_labels = get_num_labels(encoders)

    print("Number of labels per task:", num_labels)

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)

    # ----- Model -----
    model = MultiTaskDistilBERT(model_name=MODEL_NAME, num_labels=num_labels)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    best_exact_match = 0.0

    # ----- Training loop -----
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = 0
            for field in LABEL_FIELDS:
                field_loss = loss_fn(logits[field], batch[f"{field}_label"].to(DEVICE))
                loss += field_loss

            loss = loss / len(LABEL_FIELDS)  # average loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": round(loss.item(), 4)})

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, DEVICE)

        print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        print("Validation Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k:25}: {v}")

        # Save best model based on Exact Match
        if val_metrics["exact_match"] > best_exact_match:
            best_exact_match = val_metrics["exact_match"]
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"  → New best model saved (Exact Match: {best_exact_match})")

    # ----- Final Test Evaluation -----
    print("\n" + "="*50)
    print("Loading best model for Test evaluation...")
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt"))
    test_metrics = evaluate(model, test_loader, DEVICE)

    print("\n===== FINAL TEST METRICS =====")
    for k, v in test_metrics.items():
        print(f"  {k:25}: {v}")

    # Save metrics
    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel + metrics saved to → {OUTPUT_DIR}")


if __name__ == "__main__":
    train()