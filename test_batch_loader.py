
import torch

from src.training.batch_loader import load_split, get_batch

# Load the 100% training split
train_data = load_split("data/processed/train_100.bin")

# Sample a batch
x, y = get_batch(
    data=train_data,
    batch_size=4,
    block_size=256,
    device="cpu",
)

print("Dataset size:", len(train_data))
print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Input dtype:", x.dtype)
print("Target dtype:", y.dtype)

# Verify next-character alignment
assert torch.equal(x[:, 1:], y[:, :-1])

assert x.shape == (4, 256)
assert y.shape == (4, 256)
assert x.dtype == torch.long
assert y.dtype == torch.long

print("\nBatch loader validation successful!")
