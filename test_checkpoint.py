
import os
import tempfile
import torch

from src.models.model import GPT, GPTConfig
from src.training.checkpoint import save_checkpoint, load_checkpoint


# 1. Initialize a small GPT model
config = GPTConfig.from_yaml(
    "configs/small.yaml",
    vocab_size=65,
)

model = GPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 2. Perform one training step to initialize optimizer state
x = torch.randint(0, 65, (2, 256))
y = torch.randint(0, 65, (2, 256))

logits, loss = model(x, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Keep a copy of the trained weights
expected_weights = {
    key: value.clone()
    for key, value in model.state_dict().items()
}

# 3. Save and reload the checkpoint
with tempfile.TemporaryDirectory() as temp_dir:
    checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        iter_num=100,
        val_loss=1.85,
        best_val_loss=1.80,
    )

    # Create a fresh model and optimizer
    new_model = GPT(config)
    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=3e-4,
    )

    metadata = load_checkpoint(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        device="cpu",
    )

    # 4. Verify restored weights
    for key, expected in expected_weights.items():
        assert torch.equal(
            new_model.state_dict()[key],
            expected,
        ), f"Weight mismatch: {key}"

    # 5. Verify metadata and optimizer state
    assert metadata["iter_num"] == 100
    assert metadata["val_loss"] == 1.85
    assert metadata["best_val_loss"] == 1.80
    assert len(new_optimizer.state) > 0

print("Checkpoint saved successfully!")
print("Model weights restored correctly!")
print("Optimizer state restored correctly!")
print("Training metadata restored correctly!")
print("\nCheckpoint validation successful!")
