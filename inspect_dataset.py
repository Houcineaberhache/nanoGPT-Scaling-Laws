
import json
import pickle
import os

# 1. Dataset statistics
with open("data/processed/dataset_stats.json", "r") as f:
    stats = json.load(f)

print("=== DATASET STATS ===")
print(json.dumps(stats, indent=2))

# 2. Tokenizer metadata
with open("data/processed/meta.pkl", "rb") as f:
    metadata = pickle.load(f)

print("\n=== METADATA ===")
print(metadata)

# 3. Binary file sizes
print("\n=== FILE SIZES ===")

files = [
    "train_25.bin",
    "train_50.bin",
    "train_100.bin",
    "val.bin",
    "test.bin"
]

for filename in files:
    path = os.path.join("data/processed", filename)
    size_kb = os.path.getsize(path) / 1024
    print(f"{filename}: {size_kb:.2f} KB")
