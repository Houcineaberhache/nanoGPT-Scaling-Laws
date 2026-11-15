"""CLI tool to compute and display model parameter counts for nanoGPT configurations.

Usage:
    python -m src.models.count_params configs/small.yaml --vocab_size 65
"""

import argparse
import sys
from src.models.model import GPT, GPTConfig

MAX_ALLOWED_PARAMS = 10_000_000


def count_parameters(config_path: str, vocab_size: int) -> int:
    """Load config, instantiate model, print breakdown, and verify constraint."""
    print("=" * 65)
    print(f"Loading configuration from: {config_path}")
    print(f"Injecting vocab_size: {vocab_size}")
    print("=" * 65)

    config = GPTConfig.from_yaml(config_path, vocab_size=vocab_size)
    model = GPT(config)

    # 1. Embeddings breakdown
    wte_params = model.wte.weight.numel()
    wpe_params = model.wpe.weight.numel()
    embeddings_total = wte_params + wpe_params

    # 2. Transformer Blocks breakdown
    block_ln_params = 0
    block_attn_params = 0
    block_mlp_params = 0

    for block in model.h:
        # LayerNorms
        block_ln_params += block.ln_1.weight.numel()
        if block.ln_1.bias is not None:
            block_ln_params += block.ln_1.bias.numel()
        block_ln_params += block.ln_2.weight.numel()
        if block.ln_2.bias is not None:
            block_ln_params += block.ln_2.bias.numel()

        # Attention
        block_attn_params += block.attn.c_attn.weight.numel()
        if block.attn.c_attn.bias is not None:
            block_attn_params += block.attn.c_attn.bias.numel()
        block_attn_params += block.attn.c_proj.weight.numel()
        if block.attn.c_proj.bias is not None:
            block_attn_params += block.attn.c_proj.bias.numel()

        # MLP
        block_mlp_params += block.mlp.c_fc.weight.numel()
        if block.mlp.c_fc.bias is not None:
            block_mlp_params += block.mlp.c_fc.bias.numel()
        block_mlp_params += block.mlp.c_proj.weight.numel()
        if block.mlp.c_proj.bias is not None:
            block_mlp_params += block.mlp.c_proj.bias.numel()

    blocks_total = block_ln_params + block_attn_params + block_mlp_params

    # 3. Final LayerNorm
    ln_f_params = model.ln_f.weight.numel()
    if model.ln_f.bias is not None:
        ln_f_params += model.ln_f.bias.numel()

    # 4. LM Head (tied with wte)
    # Since model.lm_head.weight is model.wte.weight, additional unique params = 0
    lm_head_params = 0

    total_params = model.get_num_params()

    print("\n--- Model Architecture Hyperparameters ---")
    print(f"  block_size : {config.block_size}")
    print(f"  n_layer    : {config.n_layer}")
    print(f"  n_head     : {config.n_head}")
    print(f"  n_embd     : {config.n_embd}")
    print(f"  dropout    : {config.dropout}")
    print(f"  bias       : {config.bias}")
    print(f"  vocab_size : {config.vocab_size}")

    print("\n--- Parameter Count Breakdown ---")
    print(f"  1. Embeddings:")
    print(f"     - Token embedding (wte)    : {wte_params:>10,} (tied with lm_head)")
    print(f"     - Positional emb (wpe)     : {wpe_params:>10,}")
    print(f"     Subtotal Embeddings        : {embeddings_total:>10,}")

    print(f"\n  2. Transformer Blocks ({config.n_layer} layers):")
    print(f"     - LayerNorms (ln_1 + ln_2) : {block_ln_params:>10,}")
    print(f"     - Causal Self-Attention    : {block_attn_params:>10,}")
    print(f"     - MLP / Feed-Forward       : {block_mlp_params:>10,}")
    print(f"     Subtotal Blocks            : {blocks_total:>10,}")

    print(f"\n  3. Final Norm (ln_f)          : {ln_f_params:>10,}")
    print(f"  4. Output Head (lm_head)      : {lm_head_params:>10,} (tied to wte, 0 unique)")

    print("-" * 65)
    print(f"  TOTAL UNIQUE TRAINABLE PARAMS : {total_params:>10,} ({total_params / 1e6:.4f}M)")
    print("-" * 65)

    if total_params >= MAX_ALLOWED_PARAMS:
        print(
            f"\n[ERROR] Model parameter count ({total_params:,}) EXCEEDS the "
            f"10,000,000 parameter constraint limit!\n"
        )
        sys.exit(1)
    else:
        margin = MAX_ALLOWED_PARAMS - total_params
        print(
            f"\n[SUCCESS] Model has {total_params:,} parameters, strictly below "
            f"10,000,000 (margin: {margin:,} params remaining).\n"
        )

    return total_params


def main():
    parser = argparse.ArgumentParser(
        description="Compute and verify parameter counts for nanoGPT model configs."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file (e.g. configs/small.yaml)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=65,
        help="Vocabulary size from tokenizer (default: 65)",
    )
    args = parser.parse_args()

    count_parameters(args.config, args.vocab_size)


if __name__ == "__main__":
    main()
