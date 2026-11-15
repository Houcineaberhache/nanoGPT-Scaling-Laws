# Model Architecture & Oral Defense Guide — Student 2 (Model Engineer)

This document provides a comprehensive explanation of the model architectures, design rationale, mathematical formulations, empirical verifications, and integration interfaces implemented for the **nanoGPT Scaling Laws** project. It is authored specifically to serve as a complete study guide and reference for oral defense.

---

## 1. Overview & Role Deliverables

As **Student 2 (Model Engineer)**, my mandate was to design, implement, test, and document the character-level language model architectures from scratch in pure PyTorch (without high-level Hugging Face abstractions or direct nanoGPT copy-pasting), strictly respecting the project constraints:
1. Pure character-level Transformer decoder architecture (`nanoGPT` style).
2. Dynamic vocabulary size injection from Student 1's tokenizer (`vocab_size = 65`).
3. Two model sizes: **Small** (~0.83M params) and **Medium** (~4.80M params), both strictly below 10,000,000 parameters.
4. An analytical **Bigram Language Model** baseline with Laplace smoothing as an empirical benchmark.
5. Rigorous unit test coverage, parameter budget proofs, and strict interface compatibility.

### Delivered Files:
- `src/models/model.py`: Full Transformer implementation (`GPTConfig`, `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`, `GPT`).
- `src/models/baseline.py`: Statistical bigram baseline (`BigramLanguageModel`, `from_counts` with Laplace smoothing).
- `src/models/count_params.py`: Standalone CLI utility for validating config parameter counts against the 10M upper bound.
- `src/models/__init__.py`: Clean package exports (`GPT`, `GPTConfig`, `BigramLanguageModel`).
- `configs/small.yaml`: Hyperparameter specification for Small model (4 layers, 128 embedding dim).
- `configs/medium.yaml`: Hyperparameter specification for Medium model (6 layers, 256 embedding dim).
- `tests/test_model.py`: 11 unit tests covering forward shapes, causal masking, generation, weight tying, parameter budget, and serialization.
- `tests/test_baseline.py`: 4 unit tests covering baseline forward/loss, parameter counts, generation, and count estimation.
- `docs/architecture/interface_contract.md`: Binding interface contract for Students 3 and 4.
- `docs/architecture/model_explanation.md`: This architecture and oral defense reference.

---

## 2. Forward Pass Walkthrough & Tensor Shapes

We trace a forward pass through the **Small Model** configuration using a standard training mini-batch:
- Batch size: $B = 8$
- Sequence length: $T = 256$
- Hidden embedding dimension: $C = n_{\text{embd}} = 128$
- Number of heads: $n_{\text{head}} = 4$
- Head dimension: $d_k = C / n_{\text{head}} = 128 / 4 = 32$
- Number of layers: $n_{\text{layer}} = 4$
- Vocabulary size: $V = 65$

```
Input Tokens: idx (B=8, T=256)
  │
  ├──> wte(idx) ───────> (8, 256, 128)
  │                          +
  ├──> wpe(0..T-1) ────> (8, 256, 128) (broadcast across batch)
  │                          │
  │                     Dropout(p=0.2) ──> (8, 256, 128)
  │                          │
  ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│ Transformer Block 1..4                                      │
│                                                             │
│   x ───────────────────────┐ (Residual Stream)             │
│   │                        │                               │
│   ├──> LayerNorm 1         │                               │
│   │      │ (8, 256, 128)   │                               │
│   │      ▼                 │                               │
│   ├──> CausalSelfAttention │                               │
│   │    • Q, K, V Proj      │ (8, 256, 384)                 │
│   │    • Split 4 Heads     │ Q, K, V each (8, 4, 256, 32)  │
│   │    • Scaled Dot-Prod   │ (Q @ K.T) / sqrt(32)          │
│   │    • Causal Mask       │ Mask future with -inf         │
│   │    • Softmax + Dropout │ Attn weights (8, 4, 256, 256) │
│   │    • Attn @ V          │ Out (8, 4, 256, 32)           │
│   │    • Concat Heads      │ Reshape (8, 256, 128)         │
│   │    • Output Projection │ c_proj (8, 256, 128)          │
│   │      │                 │                               │
│   │      ▼                 │                               │
│   └──> ( + ) <─────────────┘                               │
│          │                                                 │
│   x ─────┼─────────────────┐ (Residual Stream)             │
│   │      │                 │                               │
│   ├──> LayerNorm 2         │                               │
│   │      │ (8, 256, 128)   │                               │
│   │      ▼                 │                               │
│   ├──> MLP                 │                               │
│   │    • c_fc (Linear)     │ (8, 256, 512)                 │
│   │    • GELU Activation   │ (8, 256, 512)                 │
│   │    • c_proj (Linear)   │ (8, 256, 128)                 │
│   │    • Dropout           │ (8, 256, 128)                 │
│   │      │                 │                               │
│   │      ▼                 │                               │
│   └──> ( + ) <─────────────┘                               │
│          │ (8, 256, 128)                                   │
└──────────┼──────────────────────────────────────────────────┘
           │ (Repeated through 4 blocks)
           ▼
     Final LayerNorm (ln_f) ──────> (8, 256, 128)
           │
     lm_head (tied to wte) ───────> (8, 256, 65)  [Logits]
           │
     CrossEntropyLoss(targets) ───> Scalar Tensor (Loss)
```

### Detailed Tensor Transformations Step-by-Step:
1. **Input Indices**: `idx` of shape `(8, 256)` containing character token IDs in $[0, 64]$.
2. **Token & Position Embeddings**:
   - `tok_emb = self.wte(idx)`: Table lookup of shape `(8, 256, 128)`.
   - `pos = torch.arange(0, 256)`: Position indices `(256)`.
   - `pos_emb = self.wpe(pos)`: Table lookup of shape `(256, 128)`.
   - `x = self.drop(tok_emb + pos_emb)`: Sum and dropout yield shape `(8, 256, 128)`.
3. **Transformer Block**:
   - **Pre-LN Attention**:
     - Normalized `ln1_x = self.ln_1(x)` of shape `(8, 256, 128)`.
     - Combined linear projection `c_attn(ln1_x)` produces `(8, 256, 384)`.
     - Split into $Q, K, V$, each reshaped to `(8, 4, 256, 32)`.
     - Raw attention scores: $S = \frac{Q K^T}{\sqrt{32}}$ of shape `(8, 4, 256, 256)`.
     - Lower-triangular masking: entries where `mask == 0` are filled with $-\infty$.
     - Normalization & Dropout: $A = \text{Dropout}(\text{Softmax}(S, \text{dim}=-1))$.
     - Context aggregation: $A \times V$ of shape `(8, 4, 256, 32)`.
     - Concatenation: Transpose and flatten heads to `(8, 256, 128)`.
     - Output projection: `c_proj` followed by dropout yields `(8, 256, 128)`.
     - Residual addition: $x = x + \text{attn\_out}$ of shape `(8, 256, 128)`.
   - **Pre-LN Feed-Forward (MLP)**:
     - Normalized `ln2_x = self.ln_2(x)` of shape `(8, 256, 128)`.
     - Expansion projection `c_fc(ln2_x)` expands $128 \to 512$ (`(8, 256, 512)`).
     - Nonlinear activation: `GELU()` preserves shape `(8, 256, 512)`.
     - Contraction projection `c_proj` projects $512 \to 128$ (`(8, 256, 128)`).
     - Dropout applied: `(8, 256, 128)`.
     - Residual addition: $x = x + \text{mlp\_out}$ of shape `(8, 256, 128)`.
4. **Final LayerNorm & Language Modeling Head**:
   - Normalized representation: $x = \text{ln\_f}(x)$ of shape `(8, 256, 128)`.
   - Linear projection using tied weights: `lm_head(x) = x @ wte.weight.T` yields logits `(8, 256, 65)`.
5. **Cross-Entropy Loss**:
   - Logits are flattened to `(2048, 65)` and targets to `(2048)`.
   - Standard PyTorch `F.cross_entropy` computes scalar mean negative log-likelihood.

---

## 3. Component Deep Dive & Architectural Justifications

Every architectural choice is tailored for character-level modeling on a single Shakespeare corpus under limited computational resources.

### 3.1 Manual Causal Attention vs. PyTorch Black Boxes
- **Implementation**: Written explicitly with query/key/value projections, matrix multiplication, causal masking via `register_buffer("bias", ...)`, softmax, and head concatenation.
- **Why Manual**: We deliberately avoided high-level abstractions like `torch.nn.MultiheadAttention` or opaque fused kernels (`F.scaled_dot_product_attention`). This allows line-by-line inspection, exact verification of causality, and enables our oral defense to articulate every step of the scaled dot-product attention mechanism.
- **The Causal Mask**: A lower-triangular binary matrix $M \in \{0, 1\}^{T \times T}$ where $M_{i, j} = 1 \iff j \le i$. Positions where $M_{i, j} = 0$ are replaced by $-\infty$ prior to softmax. Since $\exp(-\infty) = 0$, attention weights to future tokens are strictly zero, preventing information leakage.

### 3.2 Pre-LayerNorm vs. Post-LayerNorm
- **Original Transformer (Vaswani et al., 2017)**: Used Post-LN: $x_{l+1} = \text{LN}(x_l + \text{SubLayer}(x_l))$. In Post-LN, activations pass through normalization after every addition, causing gradients at early layers to vanish without extensive learning rate warm-up.
- **Pre-LayerNorm (our implementation)**: $x_{l+1} = x_l + \text{SubLayer}(\text{LN}(x_l))$. The residual connection forms an uninterrupted identity highway from input to output:
  $$\frac{\partial x_L}{\partial x_l} = I + \sum_{k=l}^{L-1} \frac{\partial \text{SubLayer}_k}{\partial x_k}$$
  This prevents gradient vanishing, stabilizes deep network training, and eliminates the need for finicky learning rate warmup schedules.

### 3.3 GELU vs. ReLU Activation
- **Mathematical Definition**: $\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$.
- **Justification**: ReLU ($f(x) = \max(0, x)$) has a discontinuous first derivative at $x = 0$ and completely zeroes out negative activations ("dying ReLU"). GELU provides a smooth, non-monotonic curve with non-zero curvature for small negative values, allowing small gradient updates even for inhibited units.

### 3.4 Weight Tying (`lm_head.weight = wte.weight`)
- **Mechanism**: The projection matrix mapping hidden representations back to vocabulary logits shares the exact memory tensor with the input token embedding lookup table:
  $$\text{Logits}_t = h_t \cdot W_{\text{emb}}^T$$
- **Two Major Benefits**:
  1. **Parameter Efficiency**: Avoids creating a separate $(V \times C)$ parameter matrix. In our Medium model ($V=65, C=256$), this saves $16,640$ parameters.
  2. **Geometric Regularization**: Enforces duality between input and output representations: tokens that share semantic contexts are embedded closely and receive similar output probabilities.

### 3.5 Initialization Scheme & Residual Projection Scaling
- **Linear & Embedding Layers**: Initialized with $\mathcal{N}(0, 0.02)$, following GPT-2. Zero initial biases.
- **Residual Projection Scaling**: Attention and MLP output projection weights (`c_proj.weight`) are scaled by $\frac{0.02}{\sqrt{2 \cdot n_{\text{layer}}}}$.
  - *Mathematical Derivation*: Each Transformer block adds two residual streams (attention and MLP). For $L$ layers, there are $2L$ residual additions. If each sublayer produces output with variance $\sigma^2$, the variance of the residual stream would accumulate as $\operatorname{Var}(x_L) \approx \operatorname{Var}(x_0) + 2L \sigma^2$. Scaling standard deviation by $1 / \sqrt{2L}$ dampens residual growth and ensures activation variance remains bounded as depth increases.

### 3.6 Dropout Probability of 0.2
- In standard large-scale language models trained on hundreds of billions of tokens, dropout is often 0.0 or 0.1 because data underfitting is the dominant regime.
- For our Shakespeare dataset (~1.1M characters), the model has enough capacity to memorize the text. A dropout rate of **0.2** provides necessary regularization to prevent severe overfitting.

### 3.7 Constant Context Window (`block_size = 256`)
- Both Small and Medium models share `block_size = 256`.
- *Scientific Justification*: In scaling law experiments, confounding variables must be isolated. If context length varied between models, changes in validation loss could be attributed to either parameter capacity or context availability. Keeping $T=256$ constant ensures differences in loss directly reflect model depth and width scaling.

---

## 4. Model Configurations & Hand-Calculated Parameter Formula

### 4.1 Exact Parameter Formula (`bias = False`, Tied Weights)

For a Transformer decoder with vocabulary $V$, context length $T$, embedding dimension $C$, and $L$ layers:

1. **Embeddings**:
   - Token embeddings ($wte$): $V \times C$
   - Position embeddings ($wpe$): $T \times C$
2. **Per Transformer Block**:
   - LayerNorm 1: $C$ weights (no bias when `bias=False`)
   - Causal Self-Attention:
     - QKV linear projection ($c\_attn$): $C \times (3C) = 3 C^2$
     - Output projection ($c\_proj$): $C \times C = C^2$
   - LayerNorm 2: $C$ weights
   - MLP:
     - Expansion linear ($c\_fc$): $C \times (4C) = 4 C^2$
     - Contraction linear ($c\_proj$): $(4C) \times C = 4 C^2$
   - *Total Per Block*: $2C + 12 C^2$
3. **Final LayerNorm ($ln\_f$)**:
   - $C$ weights
4. **Language Modeling Head ($lm\_head$)**:
   - $0$ additional parameters due to weight tying (`lm_head.weight = wte.weight`).

$$\mathbf{P}_{\text{total}} = (V + T + 1) \cdot C + L \cdot (2C + 12 C^2)$$

### 4.2 Step-by-Step Parameter Calculations

#### Small Model Configuration:
- $V = 65, T = 256, C = 128, L = 4$
- Embeddings & Final Norm:
  $$(65 + 256 + 1) \cdot 128 = 322 \cdot 128 = 41,216$$
- Single Block:
  $$2(128) + 12(128)^2 = 256 + 12(16,384) = 256 + 196,608 = 196,864$$
- All 4 Blocks:
  $$4 \cdot 196,864 = 787,456$$
- **Total Small Parameters**:
  $$41,216 + 787,456 = \mathbf{828,672} \quad (\approx 0.83\text{M})$$

#### Medium Model Configuration:
- $V = 65, T = 256, C = 256, L = 6$
- Embeddings & Final Norm:
  $$(65 + 256 + 1) \cdot 256 = 322 \cdot 256 = 82,432$$
- Single Block:
  $$2(256) + 12(256)^2 = 512 + 12(65,536) = 512 + 786,432 = 786,944$$
- All 6 Blocks:
  $$6 \cdot 786,944 = 4,721,664$$
- **Total Medium Parameters**:
  $$82,432 + 4,721,664 = \mathbf{4,804,096} \quad (\approx 4.80\text{M})$$

### 4.3 Architecture Comparison Table

| Component | Small Model | Medium Model |
|---|---|---|
| Number of Layers ($n_{\text{layer}}$) | 4 | 6 |
| Number of Attention Heads ($n_{\text{head}}$) | 4 | 8 |
| Hidden Embedding Dim ($n_{\text{embd}}$) | 128 | 256 |
| Head Dimension ($d_k$) | 32 | 32 |
| Context Length (`block_size`) | 256 | 256 |
| Vocabulary Size (`vocab_size`) | 65 | 65 |
| Dropout Rate | 0.2 | 0.2 |
| Bias | False | False |
| Token Embeddings ($wte$) | 8,320 | 16,640 |
| Position Embeddings ($wpe$) | 32,768 | 65,536 |
| Params per Block | 196,864 | 786,944 |
| Total Block Params | 787,456 | 4,721,664 |
| Final LayerNorm ($ln\_f$) | 128 | 256 |
| LM Head (tied) | 0 (tied) | 0 (tied) |
| **Total Parameters** | **828,672** (~0.83M) | **4,804,096** (~4.80M) |
| **Budget Utilization (< 10M)** | **8.29%** | **48.04%** |
| **Parameter Scaling Ratio** | **1.0x** | **5.7973x (~5.80x)** |

Both architectures comfortably satisfy the strict project requirement of fewer than 10,000,000 parameters.

---

## 5. Statistical Bigram Baseline & Empirical Benchmark

### 5.1 Formulation & Optimality
A Bigram language model makes a first-order Markov assumption: the probability of character $c_t$ depends solely on the immediate predecessor $c_{t-1}$:
$$P(c_1, \dots, c_T) = P(c_1) \prod_{t=2}^T P(c_t \mid c_{t-1})$$

The Maximum Likelihood Estimator (MLE) with additive **Laplace smoothing** ($\alpha$) is:
$$P(c_j \mid c_i) = \frac{C(c_i, c_j) + \alpha}{\sum_{k=1}^V (C(c_i, c_k) + \alpha)} = \frac{C(c_i, c_j) + \alpha}{N_i + \alpha \cdot V}$$
where $C(c_i, c_j)$ is the frequency of bigram $(c_i, c_j)$, $N_i = \sum_k C(c_i, c_k)$ is the count of character $c_i$, and $V = 65$.
- Stored as log-probabilities in a $(65 \times 65)$ lookup table ($4,225$ parameters).
- **Optimality**: Laplace smoothing is the optimal Bayes estimator under a symmetric Dirichlet prior over transition distributions. It prevents $-\infty$ log-probabilities (infinite perplexity) for bigrams unseen during training.

### 5.2 Verified Empirical Results (from `PROGRESS.md`)
Evaluated across the exact Shakespeare splits (with $V = 65$):
- **Training Set** (905,183 character pairs):
  - Loss: **2.4515** nats
  - Perplexity ($\text{PPL} = e^{\text{loss}}$): **11.61**
- **Validation Set** (113,147 character pairs):
  - With $\alpha = 1.0$: Loss = **2.4958** nats, $\text{PPL} = \mathbf{12.13}$
  - With $\alpha = 0.01$: Loss = **2.4969** nats, $\text{PPL} = \mathbf{12.14}$
- **Test Set** (113,149 character pairs):
  - With $\alpha = 1.0$: Loss = **2.5034** nats, $\text{PPL} = \mathbf{12.22}$

*Why $\alpha=1.0$ is superior to $\alpha=0.01$*: The vocabulary includes rare characters (e.g., uppercase letters, certain punctuation marks). A very small $\alpha=0.01$ assigns excessively small probabilities to rare valid transitions in the validation set, slightly penalizing validation cross-entropy.

### 5.3 Baseline as a Scientific Control
An untrained random model over 65 characters yields an expected loss of $\ln(65) \approx 4.1744$ nats.
The bigram baseline drops this loss to **2.4958** nats using only local character adjacency.
For Students 3 and 4, this establishes the empirical lower bound: any Transformer model that cannot beat **2.49 nats** has failed to learn long-range syntactic or semantic dependencies.

---

## 6. Verification & Unit Testing Summary

Our test suite in `tests/test_model.py` and `tests/test_baseline.py` contains 15 automated pytest tests running deterministically on CPU in ~6 seconds.

### 6.1 Test Coverage Overview
- **`tests/test_model.py` (11 tests)**:
  1. `test_forward_shapes`: Verifies logits shape `(B, T, V)`, scalar cross-entropy loss when targets are present, and `loss is None` when targets are absent.
  2. `test_sequence_length_assertion`: Confirms passing $T > \text{block\_size}$ raises `ValueError`.
  3. `test_causality_invariance`: Proves that modifying token $T$ does not alter logits at tokens $1 \dots T-1$.
  4. `test_generate_shapes_and_context_crop`: Confirms autoregressive generation preserves batch dimension, concatenates tokens correctly, and handles prompts longer than `block_size`.
  5. `test_generate_top_k_strict_membership`: Draws 200 consecutive stochastic samples from a fixed prompt and mathematically asserts every sampled token belongs to the top-$k$ set of next-token logits.
  6. `test_eval_determinism_vs_train_dropout`: Proves that two forwards in `eval()` mode are bitwise identical, whereas in `train()` mode dropout creates stochastic variations.
  7. `test_weight_tying_and_param_counting`: Verifies `model.lm_head.weight is model.wte.weight` and checks `get_num_params()` against our hand formula.
  8. `test_bias_true_runs`: Validates that `GPTConfig(bias=True)` constructs and executes cleanly without dimension mismatches.
  9. `test_from_yaml_loader_and_validation`: Tests loading real YAML configs, dynamic `vocab_size` injection, and validates exceptions for missing keys, unknown keys, conflicting `vocab_size`, and non-divisible head dimensions.
  10. `test_parameter_budget_under_10m`: Asserts both `configs/small.yaml` and `configs/medium.yaml` have $< 10,000,000$ parameters.
  11. `test_state_dict_save_and_load`: Confirms complete serialization roundtrip fidelity via `torch.save` and `load_state_dict`.
- **`tests/test_baseline.py` (4 tests)**:
  1. `test_baseline_forward_and_untrained_loss`: Checks output shapes and verifies untrained random weights produce loss close to $\ln(65) \approx 4.1744$.
  2. `test_baseline_param_count`: Validates parameter count equals $V^2 = 4,225$.
  3. `test_baseline_generate_contract`: Confirms identical `generate` contract and context cropping compatibility.
  4. `test_from_counts_estimation`: Tests empirical frequency matrix estimation and Laplace smoothing with synthetic known transitions.

### 6.2 Causality Falsification Proof
To prove that our causality test is genuinely sensitive and not a vacuous check, we executed a scientific falsification experiment:
- We temporarily patched the attention mask buffer `model.h[i].attn.bias` from its causal lower-triangular state to a matrix of all ones ($1.0$).
- When future tokens were unmasked, modifying token $T$ caused past logits at tokens $1 \dots T-1$ to change immediately.
- The assertion `torch.equal(logits1[:, :-1, :], logits2[:, :-1, :])` failed with `AssertionError: Tensor mismatch at token positions`.
- This definitively confirms the causal mask is active and strictly prevents bidirectional information leakage.

---

## 7. Integration Guide for Students 3 and 4

Students 3 (Training Engineer) and 4 (Evaluation Engineer) can copy and paste the following tested code snippets directly into their training loops and evaluation scripts.

### 7.1 Model Instantiation with Dynamic Vocabulary Size
```python
from src.models import GPT, GPTConfig, BigramLanguageModel

# 1. Load configuration and dynamically inject tokenizer vocab_size
vocab_size = 65  # Injected from tokenizer metadata (meta.pkl / dataset_stats.json)
config = GPTConfig.from_yaml("configs/small.yaml", vocab_size=vocab_size)

# 2. Instantiate Transformer model
model = GPT(config)
print(f"Model parameters: {model.get_num_params():,}")

# 3. Alternatively, instantiate the Bigram baseline with identical interface
baseline = BigramLanguageModel(vocab_size=vocab_size, block_size=config.block_size)
```

### 7.2 Training Step Forward Pass & Backpropagation
```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

# x: (B, T), y: (B, T)
x = torch.randint(0, vocab_size, (8, 256), device=device)
y = torch.randint(0, vocab_size, (8, 256), device=device)

# Forward pass computes cross-entropy loss automatically when targets are provided
logits, loss = model(x, targets=y)

# Standard optimizer step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 7.3 Autoregressive Text Generation
```python
model.eval()

# Prompt of arbitrary length (e.g., 5 tokens)
prompt_idx = torch.randint(0, vocab_size, (1, 5), device=device)

# Autoregressive generation with top-k truncation
with torch.no_grad():
    generated_idx = model.generate(
        prompt_idx,
        max_new_tokens=100,
        temperature=0.8,
        top_k=10
    )

print(f"Generated sequence shape: {generated_idx.shape}")  # (1, 105)
```

### 7.4 Checkpoint Saving and Rebuilding
```python
from dataclasses import asdict

# --- Saving (Student 3) ---
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": asdict(config),  # Serializes all GPTConfig fields as standard dict
    "step": 1000,
    "val_loss": 1.75,
}
torch.save(checkpoint, "checkpoint_small.pt")

# --- Loading & Rebuilding (Student 3 & Student 4) ---
checkpoint = torch.load("checkpoint_small.pt", map_location=device)
rebuilt_config = GPTConfig(**checkpoint["config"])
rebuilt_model = GPT(rebuilt_config).to(device)
rebuilt_model.load_state_dict(checkpoint["model_state_dict"])
rebuilt_model.eval()
```

---

## 8. Limitations & Scope Boundaries

To maintain full transparency during project evaluation, the following boundaries and environment nuances are documented:

1. **Dataset Binaries on Feature Branch**:
   - The processed dataset binary files (`train_25.bin`, `train_50.bin`, `train_100.bin`, `val.bin`, `test.bin`) and tokenizer metadata (`meta.pkl`, `dataset_stats.json`) currently reside on branch `origin/feature/data-pipeline` (commit `5eea4cf`).
   - They have not yet been merged into `main`. Full model training by Student 3 requires merging or checking out that branch.
2. **Pytest Auto-Generation Behavior**:
   - When running the global test suite (`pytest`), `tests/test_data.py` checks if `meta.pkl` exists. If missing on a fresh clone, it automatically triggers `prepare_datasets()`, downloading the Shakespeare text and creating local `.bin` files under `data/`.
   - These files are strictly `.gitignore`d and are not part of Student 2's commit or pull request.
3. **Empty Upstream `requirements.txt`**:
   - The repository's `requirements.txt` was initially empty. Environment dependencies (`torch`, `numpy`, `pyyaml`, `pytest`) were installed locally in `.venv`.
4. **Scope Separation**:
   - Student 2 verified forward/backward numerical stability on CPU using synthetic tensors, computed exact parameter budgets, and evaluated the statistical Bigram baseline on the full dataset.
   - Large-scale multi-epoch GPU training on Colab, learning rate schedules, and scaling curves are exclusively owned by Student 3 and Student 4.

---

## 9. Oral Defense FAQ (Top 10 Likely Questions)

#### Q1: Why did you choose a character-level tokenizer over Byte-Pair Encoding (BPE)?
**A**: Character-level tokenization provides a compact vocabulary ($V = 65$) where every character is meaningful, completely eliminating out-of-vocabulary (OOV) tokens. For a relatively small corpus (~1.1MB) and parameter budget (< 10M), BPE would introduce thousands of subword tokens, bloating the embedding matrix ($V \times C$) and consuming parameters that are better spent on Transformer layers.

#### Q2: Why did you implement causal self-attention manually instead of using PyTorch's `F.scaled_dot_product_attention`?
**A**: Writing the attention mechanism manually makes every step explicit: query/key/value projection, splitting into multiple heads, scaled dot-product computation, causal mask application via `masked_fill`, and softmax. This guarantees complete educational transparency, allows us to explain the exact mathematical operations to our professors, and enabled us to construct our causality falsification proof.

#### Q3: What is weight tying and what advantages does it provide?
**A**: Weight tying forces the output linear projection matrix (`lm_head.weight`) to share the exact same tensor in memory as the token embedding matrix (`wte.weight`). This saves $V \times C$ parameters ($16,640$ in Medium) and acts as an effective regularizer by mapping input tokens and output predictions to the same semantic geometric space.

#### Q4: Why are the residual projections scaled by $0.02 / \sqrt{2 \cdot n_{\text{layer}}}$?
**A**: In a Transformer with $L$ layers, each block adds two residual branches (attention and MLP), creating $2L$ successive additions to the residual stream. Without dampening, the variance of the activations grows proportionally to the depth of the network; scaling the weights of the residual output projections by $1/\sqrt{2L}$ keeps the activation variance constant at initialization throughout all layers.

#### Q5: What is the benefit of Pre-LayerNorm over Post-LayerNorm?
**A**: Pre-LayerNorm applies normalization to the inputs of the attention and MLP sublayers rather than to their sum with the residual stream. This preserves an uninterrupted identity connection through the residual highway, preventing gradients from vanishing or exploding during backpropagation and making training far more stable without complex learning rate warmups.

#### Q6: Why do Small and Medium models share the exact same `block_size = 256`?
**A**: In scaling law experiments, the goal is to isolate the effect of model capacity (number of parameters, depth, and width) on loss. If context length also varied between models, any observed performance gap would be confounded by the differing context availability; keeping `block_size=256` constant isolates parameter capacity as the sole independent variable.

#### Q7: How does your `.generate()` implementation handle prompts that exceed `block_size`?
**A**: Our `generate()` method applies context cropping (`idx_cond = idx[:, -self.config.block_size:]`) at every autoregressive generation step. This ensures that regardless of the prompt length or how many tokens have been generated, the tensor passed to the model never exceeds `block_size`, completely preventing out-of-bounds indexing in the positional embedding table.

#### Q8: Why is the Bigram Language Model an essential part of this project?
**A**: The Bigram model serves as an analytical, non-deep-learning baseline representing the maximum predictive power achievable purely from first-order character transitions. It establishes an empirical baseline loss of 2.4958 nats on our validation split; comparing our Transformer models against this baseline proves whether our neural architectures are genuinely learning higher-order linguistic patterns.

#### Q9: Why is Laplace smoothing with $\alpha = 1.0$ used in the bigram model instead of raw MLE frequencies?
**A**: Raw maximum likelihood estimation assigns zero probability to any character bigram not observed in the training split, resulting in infinite cross-entropy loss and infinite perplexity on the validation set. Additive Laplace smoothing ($\alpha=1.0$) acts as a uniform prior that guarantees every valid transition has non-zero probability, yielding stable, optimal Bayes estimation.

#### Q10: How did you verify that future tokens do not leak into earlier positions during attention?
**A**: We wrote a unit test that verifies changing any token $j > i$ leaves the logits at position $i$ bitwise identical. Furthermore, we executed a falsification test by monkeypatching the triangular mask to all ones: the test immediately failed with an `AssertionError`, proving the mask strictly enforces causality.
