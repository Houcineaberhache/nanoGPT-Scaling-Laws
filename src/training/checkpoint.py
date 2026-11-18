
"""Checkpoint utilities for saving and resuming GPT training."""

import os
import random

import numpy as np
import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    iter_num,
    val_loss,
    best_val_loss=None,
):
    """Save model, optimizer, training progress, and random states."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        # Required by the project interface contract
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "iter_num": iter_num,
        "val_loss": val_loss,

        # Additional information for resuming training
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,

        # Random states for reproducibility
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load a checkpoint and optionally restore optimizer state."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore random states
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    return {
        "iter_num": checkpoint["iter_num"],
        "val_loss": checkpoint["val_loss"],
        "best_val_loss": checkpoint.get("best_val_loss"),
    }
