"""Unit tests for character-level BigramLanguageModel baseline."""

import math
import numpy as np
import pytest
import torch
from src.models.baseline import BigramLanguageModel


def test_baseline_interface_parity():
    """Test that BigramLanguageModel adheres to the identical interface contract as GPT."""
    torch.manual_seed(42)
    vocab_size = 65
    block_size = 32
    model = BigramLanguageModel(vocab_size=vocab_size, block_size=block_size)
    model.eval()

    B, T = 4, 16
    idx = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))

    # 1. Forward with targets: produces logits (B, T, V) and scalar loss
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, vocab_size)
    assert loss is not None
    assert loss.dim() == 0
    assert math.isfinite(loss.item())

    # 2. Forward without targets: loss is None
    logits_no_targets, loss_no_targets = model(idx)
    assert logits_no_targets.shape == (B, T, vocab_size)
    assert loss_no_targets is None

    # 3. Context length limit check
    idx_too_long = torch.randint(0, vocab_size, (2, block_size + 1))
    with pytest.raises(AssertionError, match="block_size"):
        model(idx_too_long)

    # 4. Generate output shapes
    prompt = torch.randint(0, vocab_size, (2, 8))
    gen_no_topk = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=None)
    assert gen_no_topk.shape == (2, 18)

    gen_topk = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5)
    assert gen_topk.shape == (2, 18)


def test_baseline_parameter_count():
    """Test that the bigram model has exactly V * V parameters."""
    torch.manual_seed(42)
    vocab_size = 65
    model = BigramLanguageModel(vocab_size=vocab_size)
    assert model.get_num_params() == vocab_size * vocab_size
    assert model.get_num_params() == 4225


def test_from_counts_analytic_probabilities():
    """Verify that from_counts analytically matches hand-derived transition probabilities."""
    torch.manual_seed(42)
    # Synthetic token sequence with V=3
    # Bigram pairs: (0->1), (1->0), (0->1), (1->2)
    sequence = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    V = 3

    # With Laplace smoothing alpha = 1.0:
    # Row 0 counts: (0->0): 0, (0->1): 2, (0->2): 0 -> smoothed [1, 3, 1], row_sum = 5 -> [0.2, 0.6, 0.2]
    # Row 1 counts: (1->0): 1, (1->1): 0, (1->2): 1 -> smoothed [2, 1, 2], row_sum = 5 -> [0.4, 0.2, 0.4]
    # Row 2 counts: (2->0): 0, (2->1): 0, (2->2): 0 -> smoothed [1, 1, 1], row_sum = 3 -> [1/3, 1/3, 1/3]
    expected_probs = torch.tensor(
        [
            [0.2, 0.6, 0.2],
            [0.4, 0.2, 0.4],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=torch.float32,
    )

    model = BigramLanguageModel.from_counts(sequence, vocab_size=V, alpha=1.0)
    # The model weights are log-probabilities; applying softmax recovers the transition matrix
    recovered_probs = torch.softmax(model.token_embedding_table.weight, dim=-1)

    assert torch.allclose(recovered_probs, expected_probs, atol=1e-5), (
        f"Recovered probabilities do not match analytic expectation:\n{recovered_probs}\nvs\n{expected_probs}"
    )
    # Verify each row sums to 1.0
    row_sums = recovered_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(V), atol=1e-5), "Probability rows must sum to 1.0"


def test_from_counts_loss_reduction():
    """Verify that from_counts significantly reduces cross-entropy loss compared to an untrained model."""
    torch.manual_seed(42)
    V = 10
    # Generate structured synthetic data with strong transition patterns (T <= block_size)
    pattern = [0, 1, 2, 3, 4] * 40  # 200 tokens, T=199 <= block_size=256
    data = np.array(pattern, dtype=np.int64)

    untrained_model = BigramLanguageModel(vocab_size=V)
    fitted_model = BigramLanguageModel.from_counts(data, vocab_size=V, alpha=1.0)

    # Evaluate both on training sequence
    x = torch.tensor(data[:-1], dtype=torch.long).unsqueeze(0)  # (1, T)
    y = torch.tensor(data[1:], dtype=torch.long).unsqueeze(0)   # (1, T)

    with torch.no_grad():
        _, untrained_loss = untrained_model(x, targets=y)
        _, fitted_loss = fitted_model(x, targets=y)

    assert fitted_loss < untrained_loss, (
        f"Fitted loss ({fitted_loss.item():.4f}) must be lower than untrained loss ({untrained_loss.item():.4f})"
    )
    # The theoretical optimal loss for a deterministic cyclic sequence with alpha smoothing is low (< 0.5)
    assert fitted_loss.item() < 1.0, f"Fitted model loss ({fitted_loss.item():.4f}) should be low for deterministic data"
