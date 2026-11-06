# 🧠 nanoGPT — Specialized Language Modeling & Scaling Laws

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![Google Colab](https://img.shields.io/badge/Training-Google%20Colab-yellow)
![Status](https://img.shields.io/badge/Status-In%20Development-green)
![Project](https://img.shields.io/badge/Project-Deep%20Learning-purple)

## 📌 Overview

This project explores the training of a **small GPT-style Transformer from scratch** using Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) as a reference.

The model will be trained on a specialized English theatre corpus (Shakespeare) to generate theatrical dialogues and investigate how **model size** and **training data quantity** affect language modeling performance.

The project focuses on understanding Transformer architectures, training dynamics, generalization, and empirical scaling behavior under limited computational resources.

### 🎯 Objectives

- Implement and train a small GPT-style Transformer from scratch.
- Prepare and tokenize a specialized Shakespeare corpus.
- Compare different model sizes and training data quantities.
- Conduct six controlled training experiments.
- Evaluate models using cross-entropy loss and perplexity.
- Analyze generated text quality and overfitting.
- Ensure reproducibility using Google Colab and GitHub.

---

## ❓ Research Questions

1. How does model size influence language modeling performance?
2. How does training data quantity affect generalization?
3. Does increasing model size always improve generated text quality?
4. What are the trade-offs between model capacity, dataset size, and computational cost?

---

## 🏗️ System Architecture

```text
          Shakespeare Corpus
                  |
                  v
        Data Preprocessing
                  |
                  v
         Character Tokenizer
                  |
                  v
        Train / Val / Test
                  |
                  v
          nanoGPT Model
        (Transformer Decoder)
                  |
                  v
          Model Training
          (Google Colab)
                  |
          +-------+-------+
          |               |
          v               v
      Checkpoints     Training Logs
          |               |
          +-------+-------+
                  |
                  v
           Evaluation
        Loss / Perplexity
                  |
                  v
          Text Generation
                  |
                  v
        Scaling Laws Analysis
```

---

## 📁 Project Structure

```text
nanoGPT-Scaling-Laws/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── prepare.py
│   │   └── tokenizer.py
│   │
│   ├── models/
│   │   ├── model.py
│   │   └── baseline.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   └── checkpoint.py
│   │
│   └── evaluation/
│       ├── evaluate.py
│       └── generate.py
│
├── configs/
│   ├── small.yaml
│   ├── medium.yaml
│   └── experiments.yaml
│
├── notebooks/
│   └── train_colab.ipynb
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── generations/
│
├── tests/
├── docs/
│   ├── architecture/
│   └── report/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 👥 Team Organization

The project is developed by a team of four students, each responsible for a specific technical component.

| Role | Responsibilities | Deliverables |
|---|---|---|
| **Student 1 — Data Engineer** | Corpus collection, preprocessing, tokenization, dataset splitting | Clean dataset, tokenizer, data pipeline |
| **Student 2 — Model Engineer** | Transformer architecture, model configurations, baseline implementation | nanoGPT model, baseline |
| **Student 3 — Training Engineer** | Google Colab setup, training pipeline, checkpoints, experiments | Training scripts, checkpoints, logs |
| **Student 4 — Evaluation Engineer** | Loss, perplexity, generation, visualization, scaling analysis | Evaluation scripts, plots, scientific conclusions |

All team members contribute to code reviews, documentation, integration, and the final presentation.

---

## 🔬 Experimental Design

We investigate two independent variables:

- **Model size:** Small vs Medium.
- **Training data quantity:** 25%, 50%, and 100%.

### Experiment Matrix

| Run | Model | Training Data |
|---|---|---|
| E1 | Small | 25% |
| E2 | Small | 50% |
| E3 | Small | 100% |
| E4 | Medium | 25% |
| E5 | Medium | 50% |
| E6 | Medium | 100% |

**Constraints:**

- Corpus size: less than 5 MB.
- Model size: fewer than 10 million parameters.
- Training: Google Colab GPU.
- Regular checkpoint saving to Google Drive.
- Fixed validation and test sets across experiments.
- Consistent tokenizer and documented training budgets.

A statistical character-level bigram language model will serve as the baseline.

---

## 📊 Evaluation Metrics

### Cross-Entropy Loss

Measures how accurately the model predicts the next token.

### Perplexity

Measures the uncertainty of the model's predictions.

**Perplexity = exp(Cross-Entropy Loss)**

Lower perplexity indicates better predictive performance on the same evaluation corpus and tokenization.

### Qualitative Evaluation

Generated samples will be analyzed based on:

- Dialogue structure and formatting.
- Local coherence.
- Repetition.
- Character consistency.
- Potential memorization of training passages.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| nanoGPT | Transformer implementation reference |
| Google Colab | GPU training environment |
| Google Drive | Checkpoint storage |
| NumPy / Pandas | Data processing and analysis |
| Matplotlib | Training curves and visualizations |
| Git & GitHub | Version control and collaboration |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/nanoGPT-Scaling-Laws.git
cd nanoGPT-Scaling-Laws
```

Replace `YOUR_USERNAME` with the repository owner's GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Once the data pipeline is implemented:

```bash
python -m src.data.prepare
```

### 5. Train the Model

Training will be performed on Google Colab using the notebook:

`notebooks/train_colab.ipynb`

Training and evaluation commands will be documented once their implementations are completed.

> **Note:** The repository is currently under development. Initial scripts and configuration files may be placeholders.

---

## 🤝 GitHub Collaboration Workflow

Each student works on a dedicated feature branch.

| Branch | Owner |
|---|---|
| `feature/data-pipeline` | Student 1 |
| `feature/model` | Student 2 |
| `feature/training` | Student 3 |
| `feature/evaluation` | Student 4 |

### Contribution Steps

**1. Update your local repository**

```bash
git checkout main
git pull origin main
```

**2. Create your feature branch**

```bash
git checkout -b feature/your-feature
```

**3. Implement and commit your changes**

```bash
git add .
git commit -m "feat: implement your feature"
```

**4. Push your branch**

```bash
git push -u origin feature/your-feature
```

**5. Open a Pull Request**

- Open the repository on GitHub.
- Create a Pull Request targeting `main`.
- Request a review from another team member.
- Merge after approval and successful checks.

**Important:** Do not push directly to `main` after the initial project setup.

---

## 📅 Project Timeline

| Week | Milestone |
|---|---|
| Week 1 | Repository setup, corpus preparation, tokenizer |
| Week 2 | Transformer architecture, baseline, integration tests |
| Week 3 | Six training experiments on Google Colab |
| Week 4 | Evaluation, visualization, scaling analysis |
| Week 5 | Final report, reproducibility checks, presentation |

---

## 📦 Deliverables

- Reproducible GitHub repository.
- Google Colab training notebook.
- Prepared and documented Shakespeare corpus.
- Baseline model and six nanoGPT training experiments.
- Model checkpoints and training logs.
- Loss and perplexity comparisons.
- Qualitative analysis of generated text.
- Final PDF report (maximum 10 pages, excluding appendices).
- Oral presentation with individual contributions.

---

## 📚 References

- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT)
- [Andrej Karpathy — Let's Build GPT](https://github.com/karpathy/ng-video-lecture)
- [Attention Is All You Need — Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners — Brown et al.](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models — Kaplan et al.](https://arxiv.org/abs/2001.08361)

---

## 🎓 Academic Context

**Module:** Deep Learning  
**Program:** Master ADIA — Data & Artificial Intelligence Engineering  
**Institution:** Université Ibn Zohr — Faculté des Sciences d'Agadir

**Project:** nanoGPT sur corpus spécialisé et lois d'échelle

---

**Project Status:** 🚧 In Development

*This project is developed for academic and research purposes.*
