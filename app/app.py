"""
🎭 Mini Shakespeare GPT — interactive demo (Gradio)

Lets you talk to the 6 nanoGPT models trained for the project (E1 → E6):
  * "Generate" tab : one model, your own prompt and sampling settings
  * "Compare" tab  : two models, same prompt, same random seed, side by side
  * "About" tab    : the 6 experiments and what to look for

Run in Colab (from the repo root):   !python app/app.py
Run on Hugging Face Spaces:          see app/README_deploy.md  (entry point = app.py)
"""

import inspect
import os
import pickle
import random
import sys
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 0. Paths  (works in the repo, in Colab, and in a flat Hugging Face Space)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "src").exists() else HERE
for p in (HERE, ROOT):
    sys.path.insert(0, str(p))

from src.models.model import GPT, GPTConfig  # noqa: E402

DRIVE_RESULTS = Path("/content/drive/MyDrive/nanoGPT-Scaling-Laws/results")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# 1. The 6 experiments  (test perplexities come from the evaluation notebook)
# --------------------------------------------------------------------------
EXPERIMENTS = {
    #  id : (model size, data %, test perplexity)
    "E1": ("Small", "25%", 8.22),
    "E2": ("Small", "50%", 7.70),
    "E3": ("Small", "100%", 6.61),
    "E4": ("Medium", "25%", 8.09),
    "E5": ("Medium", "50%", 7.11),
    "E6": ("Medium", "100%", 5.71),
}

EXAMPLE_PROMPTS = [
    "ROMEO:",
    "JULIET:\nO Romeo, Romeo!",
    "HAMLET:\nTo be, or not to be",
    "First Citizen:\nBefore we proceed any further, hear me speak.",
    "KING RICHARD III:\n",
]


def label(exp):
    size, data, ppl = EXPERIMENTS[exp]
    return f"{exp} — {size} model · {data} of the data · test perplexity {ppl:.2f}"


CHOICES = [(label(e), e) for e in EXPERIMENTS]


# --------------------------------------------------------------------------
# 2. Tokenizer (character level, 65 characters)
# --------------------------------------------------------------------------
def _first_existing(candidates):
    for c in candidates:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


META_PATH = _first_existing([
    os.environ.get("META_PATH"),
    ROOT / "data" / "processed" / "meta.pkl",
    HERE / "data" / "processed" / "meta.pkl",
    HERE / "meta.pkl",
])
if META_PATH is None:
    raise FileNotFoundError(
        "meta.pkl not found. Expected data/processed/meta.pkl in the repo "
        "(or meta.pkl next to app.py on Hugging Face Spaces)."
    )

with open(META_PATH, "rb") as f:
    _meta = pickle.load(f)
STOI, ITOS = _meta["stoi"], _meta["itos"]


def encode(prompt):
    """Text -> token ids. Characters outside the 65-char vocabulary are dropped."""
    prompt = (prompt or "").replace("\r\n", "\n")
    kept = [c for c in prompt if c in STOI]
    dropped = sorted({c for c in prompt if c not in STOI})
    if not kept:                      # empty prompt -> start from a newline
        kept = ["\n"]
    return [STOI[c] for c in kept], dropped


def decode(ids):
    return "".join(ITOS[int(i)] for i in ids)


# --------------------------------------------------------------------------
# 3. Loading the checkpoints (lazy + cached)
# --------------------------------------------------------------------------
def _checkpoint_dirs():
    dirs = [os.environ.get("CHECKPOINT_DIR"), DRIVE_RESULTS, ROOT / "results", HERE / "checkpoints"]
    return [Path(d) for d in dirs if d is not None and Path(d).exists()]


def find_checkpoint(exp):
    for d in _checkpoint_dirs():
        for candidate in (d / exp / "best.pt", d / f"{exp}.pt"):
            if candidate.exists():
                return candidate
    raise gr.Error(
        f"Checkpoint for {exp} not found. In Colab, mount Google Drive first. "
        "On Spaces, upload the checkpoints to app/checkpoints/E1/best.pt ... E6/best.pt."
    )


_cache = {}


def load_model(exp):
    if exp in _cache:
        return _cache[exp]

    ckpt = torch.load(find_checkpoint(exp), map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    try:
        n_params = f"{model.get_num_params() / 1e6:.2f}M parameters"
    except Exception:
        n_params = f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"

    size, data, ppl = EXPERIMENTS[exp]
    info = (
        f"**{exp} — {size} model, {data} of the data** · {n_params} · "
        f"best validation loss **{ckpt.get('val_loss', float('nan')):.3f}** "
        f"(iteration {ckpt.get('iter_num', '?')}) · test perplexity **{ppl:.2f}**"
    )
    _cache[exp] = (model, info)
    return _cache[exp]


# --------------------------------------------------------------------------
# 4. Text generation
# --------------------------------------------------------------------------
def _manual_generate(model, idx, max_new_tokens, temperature, top_k):
    """Fallback sampling loop with top-k (used only if model.generate has no top_k)."""
    block_size = getattr(getattr(model, "config", None), "block_size", 256)
    for _ in range(max_new_tokens):
        cond = idx[:, -block_size:]
        out = model(cond)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx


def sample(model, idx, max_new_tokens, temperature, top_k):
    kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature)
    if top_k <= 0:                                   # same call as the evaluation notebook
        return model.generate(idx, **kwargs)
    if "top_k" in inspect.signature(model.generate).parameters:
        return model.generate(idx, top_k=top_k, **kwargs)
    return _manual_generate(model, idx, max_new_tokens, temperature, top_k)


def generate_text(exp, prompt, temperature, max_new_tokens, top_k, seed):
    model, info = load_model(exp)
    ids, dropped = encode(prompt)
    context = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        out = sample(model, context, int(max_new_tokens), float(temperature), int(top_k))

    if dropped:
        shown = " ".join(repr(c) for c in dropped)
        info += f"\n\n⚠️ Characters not in the model's vocabulary were ignored: {shown}"
    return decode(out[0].tolist()), info


def generate_one(exp, prompt, temperature, max_new_tokens, top_k, seed):
    return generate_text(exp, prompt, temperature, max_new_tokens, top_k, seed)


def generate_compare(exp_a, exp_b, prompt, temperature, max_new_tokens, top_k, seed):
    # same seed for both models -> a fair comparison
    seed = random.randint(0, 2**31 - 1) if seed is None or int(seed) < 0 else int(seed)
    text_a, info_a = generate_text(exp_a, prompt, temperature, max_new_tokens, top_k, seed)
    text_b, info_b = generate_text(exp_b, prompt, temperature, max_new_tokens, top_k, seed)
    return text_a, info_a, text_b, info_b


# --------------------------------------------------------------------------
# 5. Interface
# --------------------------------------------------------------------------
def sampling_controls():
    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature",
                            info="Low = safe and repetitive · high = creative and chaotic")
    max_tokens = gr.Slider(50, 1000, value=500, step=50, label="Length (characters)")
    top_k = gr.Slider(0, 65, value=0, step=1, label="Top-k",
                      info="0 = off. Otherwise only the k most likely characters are considered")
    seed = gr.Number(value=42, precision=0, label="Seed",
                     info="Same seed = same text. Use -1 for a different text every time")
    return temperature, max_tokens, top_k, seed


ABOUT = """
### What is this?
Six tiny GPT models trained **from scratch** on Shakespeare's plays, one **character** at a time
(vocabulary of 65 characters). We changed two things: the **model size** and the **amount of training data**.

| Run | Model | Training data | Test perplexity (lower = better) |
|---|---|---|---|
""" + "\n".join(f"| {e} | {s} | {d} | {p:.2f} |" for e, (s, d, p) in EXPERIMENTS.items()) + """

### What to look for
- Compare **E1** (Small, 25 %) with **E6** (Medium, 100 %) in the *Compare* tab: more characters look like real words and the dialogue holds together better.
- **E4** (Medium, 25 %) is barely better than **E1** (Small, 25 %): a bigger model is not better when there is too little data (it overfits).
- Every model learns the *format* of a play (a NAME in capitals followed by a colon) long before it learns grammar.

*Small ≈ 0.83 M parameters · Medium ≈ 4.80 M parameters.*
"""

with gr.Blocks(title="Mini Shakespeare GPT", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎭 Mini Shakespeare GPT\n"
        "Chat with six small Transformers trained from scratch on Shakespeare. "
        "Pick a model, write the start of a scene, and let it continue."
    )

    with gr.Tabs():
        # ---------------- Generate ----------------
        with gr.Tab("✍️ Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(CHOICES, value="E6", label="Model")
                    prompt = gr.Textbox(label="Prompt", value="ROMEO:", lines=3,
                                        info="Start like a play script: NAME: followed by a line")
                    gr.Examples([[p] for p in EXAMPLE_PROMPTS], inputs=[prompt], label="Example prompts")
                    temperature, max_tokens, top_k, seed = sampling_controls()
                    run_btn = gr.Button("🎭 Generate", variant="primary")
                with gr.Column(scale=2):
                    model_info = gr.Markdown()
                    output = gr.Textbox(label="Generated text", lines=22, show_copy_button=True)

            run_btn.click(
                generate_one,
                inputs=[model_dd, prompt, temperature, max_tokens, top_k, seed],
                outputs=[output, model_info],
            )

        # ---------------- Compare ----------------
        with gr.Tab("⚖️ Compare two models"):
            with gr.Row():
                model_a = gr.Dropdown(CHOICES, value="E1", label="Model A")
                model_b = gr.Dropdown(CHOICES, value="E6", label="Model B")
            cmp_prompt = gr.Textbox(label="Prompt (same for both)", value="ROMEO:", lines=2)
            gr.Examples([[p] for p in EXAMPLE_PROMPTS], inputs=[cmp_prompt], label="Example prompts")
            with gr.Accordion("Sampling settings (same for both)", open=False):
                c_temperature, c_max_tokens, c_top_k, c_seed = sampling_controls()
            cmp_btn = gr.Button("⚖️ Compare", variant="primary")
            with gr.Row():
                with gr.Column():
                    info_a = gr.Markdown()
                    out_a = gr.Textbox(label="Model A", lines=20, show_copy_button=True)
                with gr.Column():
                    info_b = gr.Markdown()
                    out_b = gr.Textbox(label="Model B", lines=20, show_copy_button=True)

            cmp_btn.click(
                generate_compare,
                inputs=[model_a, model_b, cmp_prompt, c_temperature, c_max_tokens, c_top_k, c_seed],
                outputs=[out_a, info_a, out_b, info_b],
            )

        # ---------------- About ----------------
        with gr.Tab("📖 About"):
            gr.Markdown(ABOUT)


if __name__ == "__main__":
    # In Colab we need a temporary public link; elsewhere (local / Spaces) a normal launch.
    demo.queue().launch(share="SPACE_ID" not in os.environ)
