# 🎭 Mini Shakespeare GPT — demo app

## A. Run it in Colab (for development and quick demos)

```python
# 1. GPU not needed (CPU is fast enough), but Drive must be mounted
from google.colab import drive
drive.mount("/content/drive")

# 2. Get the code
%cd /content
!git clone https://github.com/Houcineaberhache/nanoGPT-Scaling-Laws.git
%cd /content/nanoGPT-Scaling-Laws
!git checkout feature/demo-app        # use `main` once the PR is merged
!pip install -q "gradio>=5,<6"

# 3. Launch — a public link (valid ~72 h) appears in the output
!python app/app.py
```

The app reads the checkpoints from `MyDrive/nanoGPT-Scaling-Laws/results/E1…E6/best.pt`.

## B. Deploy it for good on Hugging Face Spaces (free)

1. Make slim checkpoints (weights only, ~3 MB for Small, ~19 MB for Medium):
   `!python app/export_slim_checkpoints.py`  → creates `app/checkpoints/E1…E6/best.pt`
2. On huggingface.co: **New Space** → SDK **Gradio** → hardware **CPU basic (free)**.
3. Upload these files to the Space (flat layout, web upload is fine):

```
app.py
requirements.txt          (remove the gradio line: the Space provides it)
meta.pkl                  (copy of data/processed/meta.pkl)
src/__init__.py
src/models/__init__.py
src/models/model.py
checkpoints/E1/best.pt … checkpoints/E6/best.pt
README.md                 (header below)
```

`README.md` header of the Space (use the version printed by `gradio.__version__` in Colab):

```
---
title: Mini Shakespeare GPT
emoji: 🎭
sdk: gradio
sdk_version: 5.x.x
app_file: app.py
---
```

4. Wait for the build (a few minutes: PyTorch is big). Share the Space URL with the teacher.

## C. Git

- Work on branch `feature/demo-app`, folder `app/`.
- **Never commit `app/checkpoints/`** — add this line to `.gitignore`: `app/checkpoints/`
