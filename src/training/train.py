
"""Training engine for character-level GPT experiments."""

import argparse
import csv
import math
import os
import pickle
import random
import time

import numpy as np
import torch

from src.models.model import GPT, GPTConfig
from src.training.batch_loader import load_split, get_batch
from src.training.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    config,
    batch_size,
    eval_iters,
    device,
):
    """Estimate average training and validation losses."""

    model.eval()
    results = {}

    for split_name, data in [
        ("train", train_data),
        ("val", val_data),
    ]:
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(
                data,
                batch_size,
                config.block_size,
                str(device),
            )

            _, loss = model(x, y)
            losses.append(loss.item())

        results[split_name] = float(np.mean(losses))

    model.train()
    return results


def train(
    config_path,
    train_path,
    val_path,
    output_dir,
    max_iters=100,
    batch_size=16,
    learning_rate=3e-4,
    eval_interval=50,
    eval_iters=10,
    log_interval=10,
    save_interval=100,
    seed=42,
    device="auto",
    resume=None,
):
    """Train a GPT model and save metrics and checkpoints."""

    # 1. Setup
    set_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    print(f"Training device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # 2. Load vocabulary metadata
    with open("data/processed/meta.pkl", "rb") as f:
        meta = pickle.load(f)

    vocab_size = meta["vocab_size"]

    # 3. Load model configuration and dataset
    config = GPTConfig.from_yaml(config_path, vocab_size)

    train_data = load_split(train_path)
    val_data = load_split(val_path)

    model = GPT(config).to(device)

    print(f"Vocabulary size: {vocab_size}")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")

    # 4. Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    start_iter = 0
    best_val_loss = float("inf")
    latest_val_loss = float("nan")

    # 5. Resume training if a checkpoint is provided
    if resume:
        metadata = load_checkpoint(
            resume,
            model,
            optimizer,
            device,
        )

        start_iter = metadata["iter_num"]

        best_val_loss = (
            metadata["best_val_loss"]
            if metadata["best_val_loss"] is not None
            else float("inf")
        )

        # Recalculate validation loss instead of trusting old metadata.
        # This also handles checkpoints created before the metadata fix.
        resume_losses = estimate_loss(
            model,
            train_data,
            val_data,
            config,
            batch_size,
            eval_iters,
            device,
        )

        latest_val_loss = resume_losses["val"]

        print(f"Resuming from iteration {start_iter}")
        print(
            f"Validation loss at resume: "
            f"{latest_val_loss:.4f}"
        )

    # 6. Prepare metrics file
    metrics_path = os.path.join(output_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)

    with open(metrics_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "iteration",
                "train_loss",
                "val_loss",
                "val_perplexity",
                "elapsed_seconds",
            ],
        )

        if write_header:
            writer.writeheader()

        # 7. Training loop
        start_time = time.time()

        for iteration in range(start_iter, max_iters):

            # Sample a mini-batch
            x, y = get_batch(
                train_data,
                batch_size,
                config.block_size,
                str(device),
            )

            # Forward and backward passes
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            completed_iter = iteration + 1

            # 8. Evaluate model
            if (
                completed_iter % eval_interval == 0
                or completed_iter == 1
                or completed_iter == max_iters
            ):
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    config,
                    batch_size,
                    eval_iters,
                    device,
                )

                train_loss = losses["train"]
                val_loss = losses["val"]

                # Keep the latest measured validation loss
                latest_val_loss = val_loss

                val_perplexity = math.exp(
                    min(val_loss, 20)
                )

                elapsed = time.time() - start_time

                writer.writerow({
                    "iteration": completed_iter,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
                    "elapsed_seconds": elapsed,
                })

                csvfile.flush()

                print(
                    f"Step {completed_iter}/{max_iters} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f} | "
                    f"Val PPL: {val_perplexity:.2f}"
                )

                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    save_checkpoint(
                        os.path.join(output_dir, "best.pt"),
                        model,
                        optimizer,
                        config,
                        completed_iter,
                        val_loss,
                        best_val_loss,
                    )

            # 9. Save latest checkpoint periodically
            if completed_iter % save_interval == 0:
                save_checkpoint(
                    os.path.join(output_dir, "latest.pt"),
                    model,
                    optimizer,
                    config,
                    completed_iter,
                    latest_val_loss,
                    best_val_loss,
                )

            if completed_iter % log_interval == 0:
                print(
                    f"Training iteration {completed_iter} "
                    f"| Batch loss: {loss.item():.4f}"
                )

        # 10. Always save final state
        save_checkpoint(
            os.path.join(output_dir, "latest.pt"),
            model,
            optimizer,
            config,
            max_iters,
            latest_val_loss,
            best_val_loss,
        )

    total_time = time.time() - start_time

    print("\nTraining completed!")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Latest validation loss: {latest_val_loss:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a character-level GPT model."
    )

    parser.add_argument(
        "--config",
        default="configs/small.yaml",
    )
    parser.add_argument(
        "--train",
        default="data/processed/train_100.bin",
    )
    parser.add_argument(
        "--val",
        default="data/processed/val.bin",
    )
    parser.add_argument(
        "--output",
        default="results/smoke_test",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
    )
