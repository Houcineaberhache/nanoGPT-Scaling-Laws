"""nanoGPT model architecture implementation in pure PyTorch.

This module implements a character-level Transformer decoder inspired by nanoGPT.
All components are written from scratch without high-level library abstractions.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hyperparameters configuring the GPT model architecture."""

    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False

    @classmethod
    def from_yaml(cls, path: str, vocab_size: int) -> "GPTConfig":
        """Load configuration from a YAML file, dynamically injecting vocab_size.

        Args:
            path: Path to the YAML configuration file.
            vocab_size: Number of unique tokens in the vocabulary (from tokenizer).

        Returns:
            Instantiated GPTConfig.

        Raises:
            ValueError: If YAML is not a dictionary, contains unknown keys, contains
                        a conflicting vocab_size, or n_embd is not divisible by n_head.
            KeyError: If any required key is missing from the YAML file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML configuration at '{path}' must be a dictionary.")

        # Check for unknown keys (catch accidental typos in YAML)
        allowed_keys = {
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "dropout",
            "bias",
            "vocab_size",
        }
        unknown_keys = sorted(list(set(data.keys()) - allowed_keys))
        if unknown_keys:
            raise ValueError(
                f"YAML configuration at '{path}' contains unknown key(s): {unknown_keys}"
            )

        # Check for required keys
        required_keys = ["block_size", "n_layer", "n_head", "n_embd"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise KeyError(
                f"YAML configuration at '{path}' is missing required key(s): {missing_keys}"
            )

        # Check for conflicting vocab_size if present in YAML
        if "vocab_size" in data and data["vocab_size"] is not None:
            if data["vocab_size"] != vocab_size:
                raise ValueError(
                    f"YAML configuration contains vocab_size={data['vocab_size']}, "
                    f"which contradicts injected vocab_size={vocab_size}."
                )

        n_embd = data["n_embd"]
        n_head = data["n_head"]
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd ({n_embd}) must be divisible by n_head ({n_head})."
            )

        return cls(
            vocab_size=vocab_size,
            block_size=data["block_size"],
            n_layer=data["n_layer"],
            n_head=n_head,
            n_embd=n_embd,
            dropout=data.get("dropout", 0.0),
            bias=data.get("bias", False),
        )


class LayerNorm(nn.Module):
    """Layer Normalization with optional learnable bias."""

    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        # Learnable scale parameter gamma: (ndim,)
        self.weight = nn.Parameter(torch.ones(ndim))
        # Optional learnable shift parameter beta: (ndim,)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, C)
        return F.layer_norm(
            x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5,
        )


class CausalSelfAttention(nn.Module):
    """Multi-head masked causal self-attention mechanism."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Key, query, value projections combined in one linear layer: (C) -> (3 * C)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection back to residual stream: (C) -> (C)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask ensures attention is strictly autoregressive (lower triangular)
        # Shape: (1, 1, block_size, block_size)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # 1. Project input to Q, K, V
        # (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x)
        # Split into separate Q, K, V tensors: each (B, T, C)
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape into multi-head format: (B, T, C) -> (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. Apply causal mask (set upper triangle to -infinity before softmax)
        # att: (B, n_head, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax to obtain attention probability weights
        # att: (B, n_head, T, T)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Weight values by attention weights
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v

        # 7. Concatenate heads back into a single vector
        # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output projection and residual dropout
        # (B, T, C) -> (B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-Forward Network (MLP) with GELU non-linearity."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expansion layer: (C) -> (4 * C)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Activation function
        self.gelu = nn.GELU()
        # Projection layer: (4 * C) -> (C)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, 4 * C)
        x = self.c_fc(x)
        # (B, T, 4 * C) -> (B, T, 4 * C)
        x = self.gelu(x)
        # (B, T, 4 * C) -> (B, T, C)
        x = self.c_proj(x)
        # (B, T, C) -> (B, T, C)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer decoder block with Pre-LayerNorm architecture."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm with residual connection for attention
        # x: (B, T, C) -> (B, T, C)
        x = x + self.attn(self.ln_1(x))
        # Pre-LayerNorm with residual connection for MLP
        # x: (B, T, C) -> (B, T, C)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """nanoGPT character-level Transformer language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding: maps character index to vector of dimension n_embd
        # Shape: (vocab_size, n_embd)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Learned positional embedding: maps position index to vector of dimension n_embd
        # Shape: (block_size, n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Input dropout
        self.drop = nn.Dropout(config.dropout)

        # Stack of n_layer Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LayerNorm before language modeling head
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Language modeling head projecting from hidden dimension to vocabulary logits
        # Shape: (n_embd, vocab_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embedding and output head
        # This dramatically reduces parameter count without sacrificing performance
        self.lm_head.weight = self.wte.weight

        # Initialize all model weights according to nanoGPT / GPT-2 standard
        self.apply(self._init_weights)

        # Apply special scaled initialization to residual output projections (c_proj)
        # Prevents activation variance from exploding with depth: std = 0.02 / sqrt(2 * n_layer)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights: Normal(0, 0.02) for Linear/Embedding, zeros for biases."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of trainable parameters in the model.

        Deduplicates tied weights (lm_head.weight == wte.weight) so they are counted once.

        Args:
            non_embedding: If True, subtracts learned positional embeddings (wpe).

        Returns:
            Total unique trainable parameters count.
        """
        seen_ptrs = set()
        total_params = 0
        for p in self.parameters():
            if p.requires_grad:
                ptr = p.data_ptr()
                if ptr not in seen_ptrs:
                    seen_ptrs.add(ptr)
                    total_params += p.numel()

        if non_embedding:
            total_params -= self.wpe.weight.numel()

        return total_params

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the Transformer model.

        Args:
            idx: Tensor of character token indices of shape (B, T), dtype torch.long.
                 Sequence length T must be <= config.block_size.
            targets: Optional ground-truth token indices of shape (B, T).

        Returns:
            Tuple of (logits, loss):
            - logits: Tensor of shape (B, T, vocab_size).
            - loss: Scalar cross-entropy loss if targets is provided, else None.
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}"
        )

        # Generate position indices [0, 1, ..., T-1]
        # Shape: (T,)
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Token embeddings + learned positional embeddings
        # tok_emb: (B, T, n_embd)
        tok_emb = self.wte(idx)
        # pos_emb: (T, n_embd) broadcasted to (B, T, n_embd)
        pos_emb = self.wpe(pos)
        # Residual stream initialization: (B, T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Pass through stack of Transformer blocks
        # x: (B, T, n_embd) -> (B, T, n_embd)
        for block in self.h:
            x = block(x)

        # Final LayerNorm
        # (B, T, n_embd) -> (B, T, n_embd)
        x = self.ln_f(x)

        # Language modeling head projection
        # (B, T, n_embd) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        # Compute cross-entropy loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten predictions: (B * T, vocab_size) and ground-truth: (B * T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens from a conditioning sequence.

        Does not alter the model's training/evaluation mode.

        Args:
            idx: Conditioning prompt tensor of token IDs, shape (B, T), dtype torch.long.
            max_new_tokens: Number of subsequent tokens to generate.
            temperature: Sampling temperature; higher = more random, lower = more deterministic.
            top_k: If set, restricts sampling to the top k highest probability tokens.

        Returns:
            Tensor of shape (B, T + max_new_tokens) containing the concatenated sequence.
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens to prevent out-of-bounds positions
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )

            # Forward the model to obtain logits for the current sequence
            # logits: (B, T_cond, vocab_size)
            logits, _ = self(idx_cond)

            # Pluck logits at the final position and apply temperature scaling
            # (B, vocab_size)
            logits = logits[:, -1, :] / temperature

            # Optionally truncate logits to top_k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask out any logit smaller than the k-th highest logit
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply softmax to convert scaled logits to normalized probabilities
            # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # Sample the next token from the probability distribution
            # idx_next: (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to the running sequence
            # (B, T + step) -> (B, T + step + 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
