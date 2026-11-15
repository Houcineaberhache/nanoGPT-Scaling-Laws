# Interface Contract — Student 2 (Model Engineer)

This document establishes the binding technical interface between the models implemented by **Student 2 (Model Engineer)** and the training/evaluation pipelines developed by **Student 3 (Training Engineer)** and **Student 4 (Evaluation Engineer)**.

---

## 1. Target Files & Model Classes

All model definitions reside in `src/models/`:
- `src/models/model.py`:
  - `GPTConfig`: Dataclass storing architecture hyperparameters with strict validation.
  - `GPT`: nanoGPT-style character-level Transformer decoder implemented in PyTorch from scratch.
- `src/models/baseline.py`:
  - `BigramLanguageModel`: Statistical character-level bigram language model implemented in PyTorch with Laplace smoothing support.
- `src/models/count_params.py`:
  - CLI utility to load configs and verify parameter counts against the < 10M budget.

---

## 2. Shared Method Signatures & Behavioral Contract

Both `GPT` and `BigramLanguageModel` implement identical public signatures, allowing Students 3 and 4 to swap models interchangeably without conditional logic.

### 2.1 `forward(idx, targets=None) -> (logits, loss)`
- **Arguments**:
  - `idx`: `torch.LongTensor` of shape `(B, T)` where $B$ is the batch size and $T$ is the sequence length ($T \le \text{block\_size}$).
  - `targets`: Optional `torch.LongTensor` of shape `(B, T)` containing the ground-truth next character indices.
- **Returns**:
  - `logits`: `torch.FloatTensor` of shape `(B, T, vocab_size)` representing unnormalized log-probabilities over the vocabulary.
  - `loss`: Scalar `torch.FloatTensor` cross-entropy loss if `targets` is provided; `None` if `targets is None`.
- **Constraint**:
  - If $T > \text{block\_size}$, raises `AssertionError` (specifically: `f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}"`).

### 2.2 `generate(idx, max_new_tokens, temperature=1.0, top_k=None) -> LongTensor`
- **Arguments**:
  - `idx`: `torch.LongTensor` of shape `(B, T)` representing conditioning prompt token indices.
  - `max_new_tokens`: Integer specifying the number of autoregressive tokens to generate.
  - `temperature`: Float controlling sharpness of sampling probabilities (default: 1.0).
  - `top_k`: Optional integer for top-$k$ truncation sampling.
- **Behavior**:
  - Autoregressively generates tokens one by one.
  - **Context Cropping**: At each iteration step, crops conditioning context to the last `block_size` tokens (`idx[:, -block_size:]`) to ensure inputs never exceed positional embedding capacity.
  - Does **not** mutate `self.training` mode during execution; runs inside `torch.no_grad()`.
- **Returns**:
  - `torch.LongTensor` of shape `(B, T + max_new_tokens)` containing prompt tokens concatenated with generated tokens.

### 2.3 `get_num_params(non_embedding: bool = False) -> int`
- Returns total number of unique trainable parameters. Deduplicates tied weights (`lm_head.weight is wte.weight`) via memory data pointer tracking (`data_ptr()`) so they are counted only once.
- If `non_embedding=True` on `GPT`, subtracts learned positional embeddings (`self.wpe.weight.numel()`). For `BigramLanguageModel`, `non_embedding` is accepted for API parity and ignored.

### 2.4 Baseline Model Specific: `BigramLanguageModel.from_counts`
- **Signature**: `BigramLanguageModel.from_counts(train_ids, vocab_size, alpha=1.0, block_size=256) -> BigramLanguageModel`
- **Arguments**:
  - `train_ids`: 1D `torch.LongTensor` or `numpy.ndarray` containing tokenized training corpus.
  - `vocab_size`: Integer vocabulary size ($V$).
  - `alpha`: Float additive Laplace smoothing parameter (default: 1.0).
  - `block_size`: Sequence context limit (default: 256).
- **Behavior**:
  - Computes transition count matrix $C(i, j)$, adds $\alpha$, normalizes rows to probabilities (posterior mean under a uniform Dirichlet prior for $\alpha=1.0$), and populates `token_embedding_table.weight` with $\log P(j \mid i)$.

---

## 3. Configuration & Runtime Vocabulary Injection

### 3.1 YAML Configuration Keys
Configurations in `configs/small.yaml` and `configs/medium.yaml` define:
- `block_size`: Maximum sequence context length, set to **256** for both Small and Medium architectures to keep context window constant during scaling experiments.
- `n_layer`: Number of Transformer blocks (Small: 4, Medium: 6).
- `n_head`: Number of attention heads (Small: 4, Medium: 8).
- `n_embd`: Embedding and hidden feature dimension (Small: 128, Medium: 256).
- `dropout`: Dropout probability, set to **0.2** for both architectures.
- `bias`: Boolean indicating whether linear layers and LayerNorm include bias, set to **false** for both architectures.

### 3.2 Dynamic `vocab_size` Rule
- `vocab_size` is **NOT** hardcoded or stored as a fixed value inside the YAML files.
- It is dynamically injected at runtime from Student 1's tokenizer metadata (`data/processed/meta.pkl` or `data/processed/dataset_stats.json`).

### 3.3 Factory Loader & Validation Rules
```python
config = GPTConfig.from_yaml(path="configs/small.yaml", vocab_size=vocab_size)
model = GPT(config)
```
`GPTConfig.from_yaml` enforces strict validation:
1. **Valid Dictionary**: Raises `ValueError` if the YAML configuration root is not a dictionary.
2. **Unknown Keys**: Raises `ValueError` if any unknown key is encountered outside the allowed set (`block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`, `vocab_size`).
3. **Missing Keys**: Raises `KeyError` if any required key (`block_size`, `n_layer`, `n_head`, `n_embd`) is missing. (`dropout` and `bias` are optional in YAML and default to `0.0` and `False` respectively, matching the `GPTConfig` dataclass defaults).
4. **Conflicting Vocabulary**: If `vocab_size` is present in the YAML file and is not `None`, it is accepted if it matches `vocab_size`, but raises `ValueError` if it contradicts the injected `vocab_size` (`data["vocab_size"] != vocab_size`).
5. **Dimension Divisibility**: Raises `ValueError` if `n_embd % n_head != 0`.

---

## 4. Checkpoints & Parameter Budget

### 4.1 Checkpoint Serialization & Loading
- Models adhere to standard PyTorch serialization via `model.state_dict()`.
- When saving checkpoints, Student 3 must serialize the configuration dictionary alongside weights so that models can be recreated without manual hyperparameter bookkeeping:

```python
from dataclasses import asdict
import torch
from src.models import GPT, GPTConfig

# --- Saving Checkpoint (Student 3) ---
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": asdict(config),  # Serializes GPTConfig fields as a dict
    "iter_num": iter_num,
    "val_loss": val_loss,
}
torch.save(checkpoint, checkpoint_path)

# --- Loading / Rebuilding Checkpoint (Student 3 & Student 4) ---
checkpoint = torch.load(checkpoint_path, map_location=device)
config = GPTConfig(**checkpoint["config"])
model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### 4.2 Parameter Counting & Strict < 10M Budget
- **Counting Formula**:
  ```python
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  ```
- **Tied Weights**: In `nanoGPT`, the token embedding weight is tied with the final linear projection head (`lm_head.weight = wte.weight`). Tied weights are counted exactly once.
- **Upper Bound**: The total trainable parameter count—including positional embeddings, layer norms, and attention projections—must strictly remain **< 10,000,000 parameters** for both Small and Medium architectures.
  - Small: **828,672 parameters** (~0.83M, 8.29% of budget)
  - Medium: **4,804,096 parameters** (~4.80M, 48.04% of budget)
  - Scaling Ratio: Medium has **5.80x** the capacity of Small.

---

## 5. Team Deliverable & Dependency Matrix

| Teammate | What Student 2 Needs From Them | What Student 2 Provides To Them |
|---|---|---|
| **Student 1** (Data Engineer) | `meta.pkl` / `dataset_stats.json` (`vocab_size = 65`), `CharTokenizer` (`encode`/`decode`), pre-tokenized `uint16` splits (`train_*.bin`, `val.bin`, `test.bin`). | Feedback on token IDs, sequence length requirements, and tokenizer roundtrip verification. |
| **Student 3** (Training Engineer) | Training environment specs, Colab GPU hardware limits, training loop integration. | `GPT` and `BigramLanguageModel` classes, `configs/small.yaml`, `configs/medium.yaml`, standard `(logits, loss)` output contract, `state_dict` compatibility. |
| **Student 4** (Evaluation Engineer) | Evaluation metrics format, generation evaluation criteria (temperature, top-$k$). | Model `.generate()` method with context cropping, cross-entropy loss computation for perplexity evaluation ($\text{PPL} = \exp(\text{loss})$). |

---

## 6. Known Issues & Operational Notes

1. **Dataset Binaries and Metadata Branch Isolation**:
   - Processed binary splits (`train_25.bin`, `train_50.bin`, `train_100.bin`, `val.bin`, `test.bin`), raw text (`data/raw/shakespeare.txt`), and tokenizer metadata (`data/processed/meta.pkl`, `data/processed/dataset_stats.json`) currently exist on remote branch `origin/feature/data-pipeline` (commit `5eea4cf`).
   - They are not yet merged into `main`. Running full training requires merging or fetching that branch.
2. **Local Pytest Auto-Generation**:
   - Running `pytest` locally triggers `tests/test_data.py`, which automatically downloads the Shakespeare corpus and builds `data/processed/` if missing. These local data files are `.gitignore`d and must not be added to commits.
3. **Empty Dependencies Specification (`requirements.txt`)**:
   - The repository's `requirements.txt` is currently an empty file. Team members must ensure standard packages (`torch`, `numpy`, `pyyaml`, `pytest`) are installed in their local environments (`.venv`).
