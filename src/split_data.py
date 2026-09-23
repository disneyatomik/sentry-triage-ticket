import json
import random
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# ====================== CONFIG ======================
INPUT_FILE = "data/raw/support_tickets.json"
OUTPUT_DIR = Path("data/processed")
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # of the remaining data after test split

random.seed(SEED)

def main():
    # Load data
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        tickets = json.load(f)

    print(f"Total tickets loaded: {len(tickets)}")

    # First split: train+val  vs  test
    train_val, test = train_test_split(
        tickets,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True
    )

    # Second split: train  vs  val
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=SEED,
        shuffle=True
    )

    print(f"\nSplit sizes:")
    print(f"  Train      : {len(train)}")
    print(f"  Validation : {len(val)}")
    print(f"  Test       : {len(test)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save splits
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved → {path}")

    
    print("\n===== Label Distribution Check (Train set) =====")
    for field in ["sentiment", "request_type", "product_area", "priority", "satisfaction"]:
        counts = Counter(t[field] for t in train)
        print(f"\n{field.upper()}:")
        for k, v in counts.most_common():
            print(f"  {k:20} {v:5} ({v/len(train)*100:.1f}%)")

    print("\nSplit completed successfully.")


if __name__ == "__main__":
    main()