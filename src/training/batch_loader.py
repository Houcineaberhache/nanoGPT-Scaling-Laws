
"""Batch loading utilities for character-level GPT training."""

import numpy as np
import torch


def load_split(path: str) -> torch.Tensor:
    """Load a uint16 tokenized dataset as a LongTensor."""
    data = np.fromfile(path, dtype=np.uint16)

    if len(data) == 0:
        raise ValueError(f"Dataset is empty: {path}")

    return torch.from_numpy(data.astype(np.int64))


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
):
    """
    Sample random input-target batches.

    x contains characters at positions [i, ..., i+block_size-1].
    y contains the corresponding next characters.
    """
    if data.ndim != 1:
        raise ValueError("Dataset must be a one-dimensional tensor.")

    if len(data) <= block_size:
        raise ValueError("Dataset must contain more tokens than block_size.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    # Random starting positions
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
    )

    # Build input and next-token targets
    x = torch.stack([
        data[i:i + block_size] for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in starts
    ])

    return x.to(device), y.to(device)
