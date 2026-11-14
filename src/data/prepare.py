import os
import json
import urllib.request
import numpy as np
from src.data.tokenizer import CharTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

RAW_FILE = os.path.join(RAW_DIR, "shakespeare.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_corpus(url: str = DATA_URL, dest_path: str = RAW_FILE) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"[INFO] Downloading Shakespeare corpus from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"[INFO] Saved raw corpus to {dest_path}")
    else:
        print(f"[INFO] Raw corpus already exists at {dest_path}")

    with open(dest_path, "r", encoding="utf-8") as f:
        return f.read()

def clean_corpus(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"

def prepare_datasets() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw_text = download_corpus()
    cleaned_text = clean_corpus(raw_text)
    total_chars = len(cleaned_text)
    file_size_mb = os.path.getsize(RAW_FILE) / (1024 * 1024)
    print(f"[INFO] Corpus size: {file_size_mb:.2f} MB ({total_chars:,} characters)")

    tokenizer = CharTokenizer.train_from_text(cleaned_text)
    meta_path = os.path.join(PROCESSED_DIR, "meta.pkl")
    tokenizer.save(meta_path)
    print(f"[INFO] Vocabulary size: {tokenizer.vocab_size} unique characters")
    print(f"[INFO] Saved tokenizer metadata to {meta_path}")

    train_end = int(total_chars * 0.80)
    val_end = int(total_chars * 0.90)

    train_text_100 = cleaned_text[:train_end]
    val_text = cleaned_text[train_end:val_end]
    test_text = cleaned_text[val_end:]

    train_100_ids = np.array(tokenizer.encode(train_text_100), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint16)
    test_ids = np.array(tokenizer.encode(test_text), dtype=np.uint16)

    n_train = len(train_100_ids)
    train_25_ids = train_100_ids[: int(n_train * 0.25)]
    train_50_ids = train_100_ids[: int(n_train * 0.50)]

    splits = {
        "train_25.bin": train_25_ids,
        "train_50.bin": train_50_ids,
        "train_100.bin": train_100_ids,
        "val.bin": val_ids,
        "test.bin": test_ids,
    }

    print("\n[INFO] Exporting binary token files (uint16):")
    for filename, arr in splits.items():
        out_path = os.path.join(PROCESSED_DIR, filename)
        arr.tofile(out_path)
        print(f"  -> {filename:<15} | {len(arr):>10,} tokens")

    stats = {
        "corpus_size_bytes": os.path.getsize(RAW_FILE),
        "corpus_size_mb": round(file_size_mb, 4),
        "total_characters": total_chars,
        "vocab_size": tokenizer.vocab_size,
        "vocabulary": list(tokenizer.stoi.keys()),
        "splits": {
            "train_25": int(len(train_25_ids)),
            "train_50": int(len(train_50_ids)),
            "train_100": int(len(train_100_ids)),
            "val": int(len(val_ids)),
            "test": int(len(test_ids)),
        },
    }
    stats_path = os.path.join(PROCESSED_DIR, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"\n[SUCCESS] Data preparation complete! Stats saved to {stats_path}")

if __name__ == "__main__":
    prepare_datasets()
