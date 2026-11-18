
import pickle
import numpy as np
import torch

from src.models.model import GPT, GPTConfig

# Reproducibility
torch.manual_seed(42)

# 1. Load dataset metadata
with open("data/processed/meta.pkl", "rb") as f:
    meta = pickle.load(f)

vocab_size = meta["vocab_size"]

# 2. Load the training data
data = np.fromfile(
    "data/processed/train_100.bin",
    dtype=np.uint16
)

data = torch.from_numpy(data.astype(np.int64))

print("Dataset loaded:", len(data), "tokens")

# 3. Load the Small GPT configuration
config = GPTConfig.from_yaml(
    "configs/small.yaml",
    vocab_size=vocab_size
)

# 4. Initialize the model
model = GPT(config)
model.eval()

print("Model initialized successfully")
print("Parameters:", model.get_num_params())

# 5. Generate a random batch
batch_size = 4
block_size = config.block_size

starts = torch.randint(
    0,
    len(data) - block_size - 1,
    (batch_size,)
)

x = torch.stack([
    data[i:i + block_size] for i in starts
])

y = torch.stack([
    data[i + 1:i + block_size + 1] for i in starts
])

# 6. Forward pass
with torch.no_grad():
    logits, loss = model(x, y)

print("\n=== BATCH VALIDATION ===")
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Logits shape:", logits.shape)
print("Initial loss:", loss.item())

print("\nForward pass successful!")
