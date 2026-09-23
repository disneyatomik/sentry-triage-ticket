import json
from pathlib import Path
from typing import Dict, List
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import joblib

# ====================== CONFIG ======================
PROCESSED_DIR = Path("data/processed")
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
LABEL_ENCODER_DIR = Path("models/label_encoders")

# Fields we want to predict
LABEL_FIELDS = ["sentiment", "request_type", "product_area", "priority", "satisfaction"]


def load_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_label_encoders(train_data: List[dict]) -> Dict[str, LabelEncoder]:
    """Fit one LabelEncoder per field using training data only."""
    encoders = {}
    LABEL_ENCODER_DIR.mkdir(parents=True, exist_ok=True)

    for field in LABEL_FIELDS:
        le = LabelEncoder()
        labels = [item[field] for item in train_data]
        le.fit(labels)
        encoders[field] = le

        # Save encoder for later use (inference + evaluation)
        joblib.dump(le, LABEL_ENCODER_DIR / f"{field}_encoder.joblib")
        print(f"Saved encoder → {field} | Classes: {list(le.classes_)}")

    return encoders


def encode_labels(data: List[dict], encoders: Dict[str, LabelEncoder]) -> List[dict]:
    """Add numeric label columns."""
    encoded = []
    for item in data:
        new_item = item.copy()
        for field in LABEL_FIELDS:
            new_item[f"{field}_label"] = int(encoders[field].transform([item[field]])[0])
        encoded.append(new_item)
    return encoded


def prepare_dataset(tokenizer) -> DatasetDict:
    # Load splits
    train_raw = load_json(PROCESSED_DIR / "train.json")
    val_raw = load_json(PROCESSED_DIR / "val.json")
    test_raw = load_json(PROCESSED_DIR / "test.json")

    # Create & save label encoders (fit only on train)
    encoders = create_label_encoders(train_raw)

    # Encode labels
    train_data = encode_labels(train_raw, encoders)
    val_data = encode_labels(val_raw, encoders)
    test_data = encode_labels(test_raw, encoders)

    # Convert to Hugging Face Dataset
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })

    # Tokenization function
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text", "ticket_id"])

    # Set format for PyTorch
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"] + [f"{f}_label" for f in LABEL_FIELDS]
    )

    return dataset, encoders


if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Preparing dataset...")
    dataset, encoders = prepare_dataset(tokenizer)

    print("\nDataset ready:")
    print(dataset)
    print("\nExample features:", dataset["train"].features)