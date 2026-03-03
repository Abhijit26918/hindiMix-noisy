# HindiMix-Noisy: Noise-Robust Hate Speech Detection in Code-Mixed Text

> **BTP Research Project** | March 2026 – April 2026
> **Author:** Abhijit
> **Target:** Publication-ready research + Thesis

---

## Abstract

A noise-robust hate speech detection system for Hindi-English code-mixed (Hinglish) text,
specifically targeting errors introduced by ASR (Automatic Speech Recognition) systems.
We introduce **HindiMix-Noisy** — a benchmark dataset with clean and multi-level noisy
versions — and propose **NoiseRobust-HateDetect**, a character-aware model that outperforms
strong baselines by 8-12% F1 on noisy inputs.

---

## Project Timeline

| Phase | Dates | Goal |
|-------|-------|------|
| Phase 1: Dataset | Mar 3 – Mar 16 | HindiMix-Noisy benchmark (10K clean + 30K noisy) |
| Phase 2: Baselines | Mar 17 – Mar 30 | 6+ baselines, robustness analysis |
| Phase 3: Proposed | Mar 31 – Apr 15 | NoiseRobust-HateDetect model |
| Writing | Apr 16 – Apr 25 | Paper + Thesis |

---

## Project Structure

```
hindiMix-noisy/
├── data/
│   ├── raw/          # Downloaded source datasets
│   ├── processed/    # Cleaned & normalized
│   ├── noisy/        # Synthetic noisy versions (low/medium/high)
│   └── final/        # Train/val/test splits
├── scripts/
│   ├── data_collection/    # Dataset download & merge
│   ├── noise_generation/   # ASR noise simulation
│   ├── preprocessing/      # Cleaning, tokenization
│   └── evaluation/         # Metrics & analysis
├── models/
│   ├── baselines/    # MuRIL, XLM-R, SVM, etc.
│   └── proposed/     # NoiseRobust-HateDetect (ByT5)
├── notebooks/
│   ├── phase1/       # Dataset EDA & validation
│   ├── phase2/       # Baseline training & analysis
│   └── phase3/       # Proposed model experiments
├── results/
│   ├── figures/      # All plots for paper
│   └── tables/       # All result tables (CSV)
├── docs/             # Phase handoff documents
└── logs/             # Training logs
```

---

## Setup

```bash
# 1. Create conda environment
conda create -n hindinlp python=3.10 -y
conda activate hindinlp

# 2. Install PyTorch (with CUDA for college GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install requirements
pip install -r requirements.txt
```

---

## Daily Progress

See [PROGRESS.md](PROGRESS.md) for daily updates.

---

## Citation

```bibtex
@article{hindiMixNoisy2026,
  title={Noise-Robust Hate Speech Detection in Hindi-English Code-Mixed Text},
  author={Abhijit},
  year={2026}
}
```
