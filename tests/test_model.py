"""Unit tests for nanoGPT model architecture (GPT and GPTConfig)."""

import math
import pytest
import torch
import yaml
from src.models.model import GPT, GPTConfig


@pytest.fixture
def tiny_config() -> GPTConfig:
    """Fixture providing a lightweight GPT configuration for fast CPU testing."""
    return GPTConfig(
        vocab_size=65,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )


def test_forward_shapes_and_loss(tiny_config):
    """Test output shapes of forward pass and loss computation."""
    torch.manual_seed(42)
    model = GPT(tiny_config)
    model.eval()

    B, T = 4, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_config.vocab_size, (B, T))

    # Forward with targets
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, tiny_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar loss
    assert math.isfinite(loss.item())

    # Forward without targets
    logits_no_targets, loss_no_targets = model(idx)
    assert logits_no_targets.shape == (B, T, tiny_config.vocab_size)
    assert loss_no_targets is None


def test_context_length_assertion(tiny_config):
    """Test that sequences exceeding block_size raise an AssertionError."""
    torch.manual_seed(42)
    model = GPT(tiny_config)
    model.eval()

    # Create sequence longer than block_size (33 > 32)
    idx = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.block_size + 1))
    with pytest.raises(AssertionError, match="block_size"):
        model(idx)


def test_causality_masking(tiny_config):
    """Test that modifying a later token strictly leaves earlier logits unchanged."""
    torch.manual_seed(42)
    model = GPT(tiny_config)
    model.eval()

    B, T = 2, 16
    idx1 = torch.randint(0, tiny_config.vocab_size, (B, T))
    idx2 = idx1.clone()
    # Perturb only the final token at position T-1
    idx2[:, -1] = (idx2[:, -1] + 1) % tiny_config.vocab_size

    with torch.no_grad():
        logits1, _ = model(idx1)
        logits2, _ = model(idx2)

    # Prefix tokens 0..T-2 must have bitwise identical logits
    diff_prefix = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max().item()
    diff_last = (logits1[:, -1, :] - logits2[:, -1, :]).abs().max().item()

    assert diff_prefix == 0.0, f"Causality violated: earlier logits changed by {diff_prefix}"
    assert diff_last > 0.0, "Perturbed token must change logits at its own position"


def test_generate_behavior(tiny_config):
    """Test generation output shapes, top_k sampling, context cropping, and mode preservation."""
    torch.manual_seed(42)
    model = GPT(tiny_config)

    # 1. Output shapes with and without top_k
    prompt = torch.randint(0, tiny_config.vocab_size, (2, 8))
    gen_no_topk = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=None)
    assert gen_no_topk.shape == (2, 18)

    gen_topk = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5)
    assert gen_topk.shape == (2, 18)

    # 2. Context cropping: prompt + new tokens exceeding block_size (10 + 30 = 40 > 32)
    gen_cropped = model.generate(prompt, max_new_tokens=30, temperature=1.0)
    assert gen_cropped.shape == (2, 38)

    # 3. top_k set verification: sample 200 tokens and ensure all belong to top-k logits
    single_prompt = torch.tensor([[5, 12, 23]], dtype=torch.long)
    k = 5
    with torch.no_grad():
        logits, _ = model(single_prompt)
        last_logits = logits[0, -1, :]
        topk_indices = set(torch.topk(last_logits, k=k).indices.tolist())

    samples = [
        model.generate(single_prompt, max_new_tokens=1, top_k=k)[0, -1].item()
        for _ in range(200)
    ]
    for s in samples:
        assert s in topk_indices, f"Sampled token {s} not in top-{k} set {topk_indices}"

    # 4. Mode preservation: generate must not alter model.training state
    model.train()
    _ = model.generate(prompt, max_new_tokens=2)
    assert model.training is True, "generate() altered model.training in train mode"

    model.eval()
    _ = model.generate(prompt, max_new_tokens=2)
    assert model.training is False, "generate() altered model.training in eval mode"


def test_eval_determinism_vs_train_dropout(tiny_config):
    """Test that eval mode is deterministic and train mode is stochastic with dropout > 0."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=tiny_config.vocab_size,
        block_size=tiny_config.block_size,
        n_layer=tiny_config.n_layer,
        n_head=tiny_config.n_head,
        n_embd=tiny_config.n_embd,
        dropout=0.2,
        bias=False,
    )
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 16))

    # Eval mode: bitwise identical outputs
    model.eval()
    with torch.no_grad():
        out1, _ = model(idx)
        out2, _ = model(idx)
    assert torch.equal(out1, out2), "Eval mode forward passes must be strictly deterministic"

    # Train mode: stochastic outputs due to active dropout
    model.train()
    out3, _ = model(idx)
    out4, _ = model(idx)
    assert not torch.equal(out3, out4), "Train mode with dropout must be stochastic across runs"


def test_weight_tying_and_param_formula(tiny_config):
    """Test that lm_head is tied to wte and parameter count matches the analytical formula."""
    torch.manual_seed(42)
    model = GPT(tiny_config)

    # Weight tying object identity
    assert model.lm_head.weight is model.wte.weight, "lm_head.weight must be identical to wte.weight"

    # Analytical formula for bias=False:
    # P = (vocab_size + block_size + 1) * n_embd + n_layer * (2 * n_embd + 12 * n_embd^2)
    V = tiny_config.vocab_size
    T = tiny_config.block_size
    C = tiny_config.n_embd
    L = tiny_config.n_layer
    expected_params = (V + T + 1) * C + L * (2 * C + 12 * (C ** 2))

    assert model.get_num_params() == expected_params, (
        f"Parameter count mismatch: {model.get_num_params()} != {expected_params}"
    )


def test_bias_option(tiny_config):
    """Test that bias=True builds, runs forward, and backpropagates successfully."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=tiny_config.vocab_size,
        block_size=tiny_config.block_size,
        n_layer=tiny_config.n_layer,
        n_head=tiny_config.n_head,
        n_embd=tiny_config.n_embd,
        dropout=0.0,
        bias=True,
    )
    model = GPT(config)
    model.train()

    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx, targets=targets)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    loss.backward()
    assert model.wte.weight.grad is not None


def test_from_yaml_validation(tmp_path):
    """Test YAML loading of real configs and validation error handling."""
    torch.manual_seed(42)

    # 1. Load real configs
    cfg_small = GPTConfig.from_yaml("configs/small.yaml", vocab_size=65)
    assert cfg_small.vocab_size == 65
    assert cfg_small.n_layer == 4
    assert cfg_small.n_embd == 128

    cfg_medium = GPTConfig.from_yaml("configs/medium.yaml", vocab_size=65)
    assert cfg_medium.vocab_size == 65
    assert cfg_medium.n_layer == 6
    assert cfg_medium.n_embd == 256

    # 2. Reject missing required keys
    bad_yaml_missing = tmp_path / "missing.yaml"
    bad_yaml_missing.write_text("block_size: 128\nn_layer: 2\n")
    with pytest.raises(KeyError, match="missing required key"):
        GPTConfig.from_yaml(str(bad_yaml_missing), vocab_size=65)

    # 3. Reject conflicting vocab_size
    bad_yaml_vocab = tmp_path / "vocab.yaml"
    bad_yaml_vocab.write_text(
        "block_size: 128\nn_layer: 2\nn_head: 2\nn_embd: 64\nvocab_size: 50\n"
    )
    with pytest.raises(ValueError, match="contradicts injected vocab_size"):
        GPTConfig.from_yaml(str(bad_yaml_vocab), vocab_size=65)

    # 4. Reject unknown keys (typos)
    bad_yaml_typo = tmp_path / "typo.yaml"
    bad_yaml_typo.write_text(
        "block_size: 128\nn_layer: 2\nn_head: 2\nn_embd: 64\ntypo_key: 123\n"
    )
    with pytest.raises(ValueError, match="unknown key"):
        GPTConfig.from_yaml(str(bad_yaml_typo), vocab_size=65)

    # 5. Reject n_embd not divisible by n_head
    bad_yaml_div = tmp_path / "div.yaml"
    bad_yaml_div.write_text(
        "block_size: 128\nn_layer: 2\nn_head: 5\nn_embd: 64\n"
    )
    with pytest.raises(ValueError, match="must be divisible"):
        GPTConfig.from_yaml(str(bad_yaml_div), vocab_size=65)


def test_parameter_budget_constraint():
    """Verify that both Small and Medium models have fewer than 10,000,000 parameters."""
    cfg_small = GPTConfig.from_yaml("configs/small.yaml", vocab_size=65)
    model_small = GPT(cfg_small)
    assert model_small.get_num_params() < 10_000_000
    assert model_small.get_num_params() == 828_672

    cfg_medium = GPTConfig.from_yaml("configs/medium.yaml", vocab_size=65)
    model_medium = GPT(cfg_medium)
    assert model_medium.get_num_params() < 10_000_000
    assert model_medium.get_num_params() == 4_804_096


def test_state_dict_roundtrip(tiny_config, tmp_path):
    """Test that state_dict serialization and loading preserves exact model output."""
    torch.manual_seed(42)
    model1 = GPT(tiny_config)
    model1.eval()

    ckpt_path = tmp_path / "model.pt"
    torch.save(model1.state_dict(), ckpt_path)

    model2 = GPT(tiny_config)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model2.eval()

    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    with torch.no_grad():
        logits1, _ = model1(idx)
        logits2, _ = model2(idx)

    assert torch.equal(logits1, logits2), "Loaded model must produce bitwise identical logits"


def test_backward_pass_gradients(tiny_config):
    """Test that a backward pass produces valid, non-None gradients for all trainable parameters."""
    torch.manual_seed(42)
    model = GPT(tiny_config)
    model.train()

    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    targets = torch.randint(0, tiny_config.vocab_size, (2, 16))

    logits, loss = model(idx, targets=targets)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"
