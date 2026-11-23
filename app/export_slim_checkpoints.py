"""
Make small checkpoints (weights + config only, no optimizer state) for deployment.

Usage (in Colab, after mounting Google Drive):
    !python app/export_slim_checkpoints.py
    # or with explicit paths:
    !python app/export_slim_checkpoints.py <results_dir> <output_dir>

Output: app/checkpoints/E1/best.pt ... E6/best.pt   (DO NOT commit these to GitHub)
"""
import sys
from pathlib import Path

import torch

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/nanoGPT-Scaling-Laws/results")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "app/checkpoints")

for exp in ["E1", "E2", "E3", "E4", "E5", "E6"]:
    ckpt = torch.load(SRC / exp / "best.pt", map_location="cpu", weights_only=False)
    slim = {k: ckpt[k] for k in ("model_state_dict", "config", "iter_num", "val_loss")}
    out = DST / exp / "best.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)
    print(f"{exp}: {out}  ({out.stat().st_size / 1024**2:.1f} MB)")
