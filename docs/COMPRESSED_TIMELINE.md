# Compressed Timeline: March 3 → April 25, 2026

**53 days. No days off. Publication target.**

---

## PHASE 1: Dataset Creation (Mar 3 – Mar 16, 14 days)

| Day | Date | Task | Script/File |
|-----|------|------|-------------|
| 1 | Mar 3 | Project setup, git, conda env | ✅ Done |
| 2 | Mar 4 | Download HASOC 2019/2020/2021 datasets | `01_download_datasets.py` |
| 3 | Mar 5 | Download SemEval, Davidson, explore & merge | `02_explore_and_merge.py` |
| 4 | Mar 6 | Clean & normalize merged dataset | `02_explore_and_merge.py` |
| 5 | Mar 7 | Implement noise injection (char, word, phonetic) | `03_add_noise.py` |
| 6 | Mar 8 | Generate 3-level noisy versions (low/med/high) | `03_add_noise.py` |
| 7 | Mar 9 | Create train/val/test splits | `04_create_splits.py` |
| 8 | Mar 10 | Dataset EDA notebook (plots, stats) | `notebooks/phase1/EDA.ipynb` |
| 9 | Mar 11 | Real ASR transcriptions via Whisper (500 samples) | `05_whisper_transcribe.py` |
| 10 | Mar 12 | Error taxonomy annotation (500 samples) | `notebooks/phase1/error_taxonomy.ipynb` |
| 11 | Mar 13 | Dataset validation & quality checks | `06_validate_dataset.py` |
| 12 | Mar 14 | Write Phase 1 handoff document | `docs/phase1_handoff.md` |
| 13 | Mar 15 | Buffer / catch-up / fix issues | — |
| 14 | Mar 16 | Push complete Phase 1 to GitHub ✅ | `daily_push.sh` |

**Phase 1 Deliverable:** `data/final/` with 10K+ clean + 30K noisy samples

---

## PHASE 2: Baseline Experiments (Mar 17 – Mar 30, 14 days)

| Day | Date | Task | Script/File |
|-----|------|------|-------------|
| 15 | Mar 17 | Classical baselines: TF-IDF + SVM/LR | `models/baselines/classical.py` |
| 16 | Mar 18 | Fine-tune MuRIL (Google's multilingual BERT for Indian langs) | `models/baselines/muril_trainer.py` |
| 17 | Mar 19 | Fine-tune XLM-R | `models/baselines/xlmr_trainer.py` |
| 18 | Mar 20 | Fine-tune mBERT | `models/baselines/mbert_trainer.py` |
| 19 | Mar 21 | Character-level CNN baseline | `models/baselines/char_cnn.py` |
| 20 | Mar 22 | Evaluate all baselines on clean test | `scripts/evaluation/evaluate.py` |
| 21 | Mar 23 | Evaluate all baselines on noisy test sets | `scripts/evaluation/evaluate_noisy.py` |
| 22 | Mar 24 | Robustness analysis (degradation curves) | `notebooks/phase2/robustness_analysis.ipynb` |
| 23 | Mar 25 | Error categorization (which errors hurt most?) | `notebooks/phase2/error_analysis.ipynb` |
| 24 | Mar 26 | Generate all result tables (CSV) | `results/tables/` |
| 25 | Mar 27 | Generate all figures (degradation plots) | `results/figures/` |
| 26 | Mar 28 | Write Phase 2 handoff + results chapter draft | `docs/phase2_handoff.md` |
| 27 | Mar 29 | Buffer / re-runs if needed | — |
| 28 | Mar 30 | Push complete Phase 2 to GitHub ✅ | `daily_push.sh` |

**Phase 2 Deliverable:** Baseline F1 scores on all noise levels, robustness analysis

---

## PHASE 3: Proposed Solution (Mar 31 – Apr 15, 16 days)

| Day | Date | Task | Script/File |
|-----|------|------|-------------|
| 29 | Mar 31 | Design NoiseRobust-HateDetect architecture | `models/proposed/architecture.md` |
| 30 | Apr 1 | Implement ByT5 backbone + classification head | `models/proposed/noise_robust_model.py` |
| 31 | Apr 2 | Implement phonetic feature extractor | `models/proposed/phonetic_features.py` |
| 32 | Apr 3 | Implement noise-augmented training loop | `models/proposed/trainer.py` |
| 33 | Apr 4 | Train on college GPU (initial run) | `models/proposed/train.py` |
| 34 | Apr 5 | Evaluate vs baselines | `scripts/evaluation/compare_all.py` |
| 35 | Apr 6 | Hyperparameter tuning (LR, batch size, aug ratio) | Colab runs |
| 36 | Apr 7 | Ablation study: remove each component one-by-one | `notebooks/phase3/ablation.ipynb` |
| 37 | Apr 8 | Real ASR test set evaluation | `scripts/evaluation/eval_real_asr.py` |
| 38 | Apr 9 | Statistical significance tests (paired t-test) | `notebooks/phase3/significance_tests.ipynb` |
| 39 | Apr 10 | Final result tables + figures for paper | `results/` |
| 40 | Apr 11 | Error case study (where does our model still fail?) | `notebooks/phase3/error_cases.ipynb` |
| 41 | Apr 12 | Buffer / improvements | — |
| 42 | Apr 13 | Push Phase 3 code + results to GitHub ✅ | `daily_push.sh` |
| 43 | Apr 14 | Write model card / HuggingFace upload | — |
| 44 | Apr 15 | Code cleanup, documentation | — |

**Phase 3 Deliverable:** NoiseRobust-HateDetect model, +8-12% F1 over best baseline

---

## WRITING PHASE (Apr 16 – Apr 25, 10 days)

| Day | Date | Task |
|-----|------|------|
| 45 | Apr 16 | Write Abstract + Introduction (paper) |
| 46 | Apr 17 | Write Related Work section |
| 47 | Apr 18 | Write Dataset section (Phase 1) |
| 48 | Apr 19 | Write Experiments section (Phase 2 + 3) |
| 49 | Apr 20 | Write Results + Analysis section |
| 50 | Apr 21 | Write Conclusion + Future Work |
| 51 | Apr 22 | Thesis: Chapters 1-3 (Intro, Literature, Dataset) |
| 52 | Apr 23 | Thesis: Chapters 4-5 (Experiments, Results) |
| 53 | Apr 24 | Final proofreading + formatting (both paper + thesis) |
| —  | Apr 25 | **SUBMISSION DEADLINE** ✅ |

---

## Target Venues for Paper

| Venue | Deadline | Type |
|-------|----------|------|
| ACL 2026 (findings) | ~Feb 2026 (missed) | Top tier |
| EMNLP 2026 | ~May 2026 | **Target** |
| COLING 2026 | ~Apr 2026 | **Target** |
| ICON 2026 (Indian) | ~Aug 2026 | Backup |

**Primary target: EMNLP 2026** — deadline likely around May-June 2026. Your timeline fits perfectly.

---

## Key Success Metrics

| Metric | Target |
|--------|--------|
| Clean test F1 | > 80% |
| Medium noise F1 (best baseline) | ~65-70% |
| Medium noise F1 (our model) | > 78% |
| Improvement over baseline | +8-12% F1 |
| Statistical significance | p < 0.05 |
