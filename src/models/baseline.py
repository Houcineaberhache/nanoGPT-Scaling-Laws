"""Statistical character-level Bigram Language Model baseline.

This module provides a pure bigram language model implementing the identical
forward and generate contract as GPT, allowing seamless benchmarking.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Character-level bigram language model.

    The model maps each character to a categorical logit distribution over the
    next character via a single lookup table: token_embedding_table[i, j]
    represents the unnormalized logit of predicting character j given character i.
    """

    def __init__(self, vocab_size: int, block_size: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        # Bigram transition table: (vocab_size, vocab_size)
        # Entry (i, j) corresponds to logit of predicting character j following character i
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        # Initialize transition weights
        torch.nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of trainable parameters in the model.

        Args:
            non_embedding: Kept for API parity with GPT; ignored for bigram model.

        Returns:
            Total trainable parameters count (vocab_size * vocab_size).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the bigram language model.

        Args:
            idx: Tensor of character token indices of shape (B, T), dtype torch.long.
                 Sequence length T must be <= block_size for interface parity.
            targets: Optional ground-truth token indices of shape (B, T).

        Returns:
            Tuple of (logits, loss):
            - logits: Tensor of shape (B, T, vocab_size).
            - loss: Scalar cross-entropy loss if targets is provided, else None.
        """
        B, T = idx.size()
        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, block_size is {self.block_size}"
        )

        # Lookup next-token logits for each token in the sequence
        # (B, T) -> (B, T, vocab_size)
        logits = self.token_embedding_table(idx)

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

        Maintains identical signature, context cropping, and sampling behavior as GPT.
        Does not alter the model's training/evaluation mode.

        Args:
            idx: Conditioning prompt tensor of token IDs, shape (B, T), dtype torch.long.
            max_new_tokens: Number of subsequent tokens to generate.
            temperature: Sampling temperature; higher = more random, lower = more deterministic.
            top_k: If set, restricts sampling to the top k highest probability tokens.

        Returns:
            Tensor of shape (B, T + max_new_tokens) containing prompt + generated tokens.
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (for interface parity with GPT)
            idx_cond = (
                idx
                if idx.size(1) <= self.block_size
                else idx[:, -self.block_size :]
            )

            # Forward the model to obtain logits for the current sequence
            # logits: (B, T_cond, vocab_size)
            logits, _ = self(idx_cond)

            # Pluck logits at the final position and scale by temperature
            # (B, vocab_size)
            logits = logits[:, -1, :] / temperature

            # Optionally truncate logits to top_k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Compute normalized next-token probabilities
            # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # Sample next token from categorical distribution
            # idx_next: (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to running sequence
            # (B, T + step) -> (B, T + step + 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @classmethod
    def from_counts(
        cls,
        train_ids: Union[np.ndarray, torch.Tensor],
        vocab_size: int,
        alpha: float = 1.0,
        block_size: int = 256,
    ) -> "BigramLanguageModel":
        """Construct the optimal statistical bigram language model directly from corpus counts.

        Mathematical Foundation:
        Under a first-order Markov assumption, the optimal Maximum Likelihood Estimate (MLE)
        of transition probabilities is P_MLE(x_{t+1}=j | x_t=i) = C(i, j) / C(i).
        To avoid zero probabilities for unobserved bigrams (which yield infinite cross-entropy),
        we place a symmetric Dirichlet prior Dir(alpha) over each row, yielding the
        posterior mean under a uniform Dirichlet prior (Laplace smoothing):
            P_smooth(j | i) = (C(i, j) + alpha) / (sum_k C(i, k) + V * alpha)

        Since softmax(log P_smooth(i, :)) == P_smooth(i, :), directly loading the log-probabilities
        into the embedding table yields the theoretically exact optimal bigram model without
        requiring gradient descent training.

        Args:
            train_ids: 1D array or tensor containing sequence of training token IDs.
            vocab_size: Size of character vocabulary V.
            alpha: Additive Laplace smoothing parameter (default: 1.0).
            block_size: Context length limit for interface parity (default: 256).

        Returns:
            Pre-computed, optimal BigramLanguageModel instance.
        """
        # Convert to 1D numpy array if tensor
        if isinstance(train_ids, torch.Tensor):
            arr = train_ids.detach().cpu().numpy()
        else:
            arr = np.asarray(train_ids)

        arr = arr.reshape(-1)
        assert len(arr) >= 2, "train_ids must contain at least 2 tokens to form bigrams."

        # Consecutive pairs: x_t -> x_{t+1}
        x_t = arr[:-1].astype(np.int64)
        x_next = arr[1:].astype(np.int64)

        # Fast vectorized 2D histogram tallying
        flat_indices = x_t * vocab_size + x_next
        counts = np.bincount(flat_indices, minlength=vocab_size * vocab_size).reshape(
            vocab_size, vocab_size
        )

        # Apply Laplace smoothing and row-normalize to valid probability distribution
        smoothed_counts = counts.astype(np.float64) + float(alpha)
        # Sum across destination vocabulary (axis=1): (vocab_size, 1)
        row_totals = smoothed_counts.sum(axis=-1, keepdims=True)
        # Transition probability matrix: (vocab_size, vocab_size)
        probs = smoothed_counts / row_totals

        # Compute log-probabilities (exact logits for cross-entropy / softmax)
        log_probs = np.log(probs)

        # Instantiate model and copy log-probabilities into embedding weights
        model = cls(vocab_size=vocab_size, block_size=block_size)
        with torch.no_grad():
            model.token_embedding_table.weight.copy_(
                torch.from_numpy(log_probs).float()
            )

        return model
