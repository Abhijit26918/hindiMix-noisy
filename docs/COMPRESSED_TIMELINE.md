# HindiMix-Noisy: Revised Timeline (Apr 6 → Apr 25, 2026)

**19 days remaining. GPU access: Apr 10-13 (RTX 4000). Deadline: Apr 25.**

---

## CURRENT STATUS (Apr 6)

| Component | Status |
|---|---|
| Data pipeline (scripts 01-04) | DONE |
| Dataset: 112,542 samples, 21,450 hi-en | DONE |
| HASOC 2021 parsed (2,819 labeled Hinglish) | DONE |
| TF-IDF SVM/LR (all noise levels) | DONE — `results/tables/cpu_results/` |
| CharCNN (all noise levels) | DONE — `results/tables/cpu_results/` |
| mBERT (all noise levels) | DONE — `results/tables/Gpu_results/` |
| XLM-R (all noise levels) | RUNNING — Kaggle |
| MuRIL (all noise levels) | RUNNING — Kaggle |
| ByT5 standard (all noise levels) | RUNNING — Kaggle (second account) |
| NoiseBridge (PWNIC + GRL) | CODED — ready to train Thu |

---

## ARCHITECTURE: NoiseBridge

**Files:** `models/proposed/noisebridge.py` + `models/proposed/train_noisebridge.py`

**4 components working together:**
1. ByT5 byte-level encoder — no OOV problem on noisy Hinglish
2. Noise-Aware Attention gate — learned token-level weighting
3. PWNIC loss — Phonetic-Weighted Noise-Invariant Contrastive (novel)
4. Gradient Reversal Layer — adversarial noise disentanglement (novel application)

**Full loss:**
```
L_total = L_CE + 0.5*L_PWNIC - 0.3*L_adv + 0.1*L_aux
```

**PWNIC (mathematical novelty):**
```
L_PWNIC = -sum_i phi_i * log[exp(sim(z_c_i, z_n_i)/t) / sum_k exp(sim(z_c_i, z_k)/t)]
phi_i = 1 - JaroWinkler(x_clean_i, x_noisy_i)  in [0,1]
```
Standard InfoNCE weighted by phonetic dissimilarity. Pushes harder when noisy version
sounds more different. First use of phonetic weighting in contrastive NLP.

**GRL (novel application to noise):**
- Forward: identity
- Backward: multiply gradient by -lambda (reversal)
- Lambda annealed via DANN schedule: lambda(p) = 2/(1+exp(-10p)) - 1
- Forces encoder to make representation UNINFORMATIVE w.r.t. noise level
- Result: hate semantics preserved, noise characteristics removed from feature space

---

## REMAINING WORK

### Apr 6 — Tonight
- [ ] Wait for Kaggle: XLM-R, MuRIL, ByT5 standard results
- [ ] Download result JSONs to `results/tables/Gpu_results/`

### Apr 7 (Wed)
- [ ] NoiseBridge + XLM-R backbone on Kaggle:
  `python models/proposed/train_noisebridge.py --encoder xlm-roberta-base --noise all --fp16`
- [ ] NoiseBridge + mBERT backbone on Kaggle:
  `python models/proposed/train_noisebridge.py --encoder bert-base-multilingual-cased --noise all --fp16`

### Apr 8 (Thu) — RTX 4000
- [ ] Setup: conda env + torch cu121 + transformers + jellyfish
- [ ] Train NoiseBridge + ByT5 (MAIN CLAIM):
  `python models/proposed/train_noisebridge.py --encoder google/byt5-small --noise all --fp16 --batch_size 4`
- [ ] Train NoiseBridge + MuRIL:
  `python models/proposed/train_noisebridge.py --encoder google/muril-base-cased --noise all --fp16 --batch_size 16`

### Apr 9 (Fri)
- [ ] Collect all results
- [ ] Generate master results table
- [ ] Run ablation study (remove GRL / remove PWNIC / remove both)

### Apr 10-11 (Sat-Sun)
- [ ] Paper writing: Abstract, Introduction, Dataset, Model sections
- [ ] Generate all figures: degradation curves, ablation bars

### Apr 12-13 (Mon-Tue)
- [ ] Paper writing: Results, Analysis, Related Work, Conclusion

### Apr 14-18
- [ ] Full paper draft, proofread, format (EMNLP/COLING style)

### Apr 19-24
- [ ] BTP thesis (5 chapters from paper sections)

### Apr 25 — DEADLINE

---

## MODELS CHECKLIST

| Model | Backbone | Status | Where |
|---|---|---|---|
| TF-IDF SVM | — | DONE | cpu_results/ |
| TF-IDF LR | — | DONE | cpu_results/ |
| CharCNN | — | DONE | cpu_results/ |
| mBERT standard | mBERT | DONE | Gpu_results/ |
| XLM-R standard | XLM-R | Running | Kaggle |
| MuRIL standard | MuRIL | Running | Kaggle |
| ByT5 standard | ByT5 | Running | Kaggle |
| NoiseBridge | mBERT | Todo Wed | Kaggle |
| NoiseBridge | XLM-R | Todo Wed | Kaggle |
| NoiseBridge | MuRIL | Todo Thu | RTX 4000 |
| NoiseBridge | ByT5 | Todo Thu | RTX 4000 |

---

## PAPER STORY (3 contributions)

1. **HindiMix-Noisy benchmark** — first systematic ASR-noise evaluation for Hinglish hate speech
   (6 models x 4 noise levels, publicly released)

2. **PWNIC loss** — phonetically-grounded variable-pressure contrastive objective.
   Novel: phonetic dissimilarity as contrastive weight. Not done before in NLP.

3. **NoiseBridge** — PWNIC (output space) + GRL disentanglement (feature space).
   Attacks noise robustness at two levels simultaneously.

Target result: NoiseBridge reduces degradation (clean F1 - high F1) by X% vs best baseline.

---

## TARGET VENUES

| Venue | Deadline | Fit |
|---|---|---|
| EMNLP 2026 | ~May 2026 | Primary |
| COLING 2026 | ~Apr 2026 | Primary |
| ICON 2026 | ~Aug 2026 | Safe backup |

---

## KEY FILES

| File | Purpose |
|---|---|
| `models/proposed/noisebridge.py` | NoiseBridge model + PWNIC loss + GRL |
| `models/proposed/train_noisebridge.py` | Full trainer (all encoders + noise levels) |
| `models/proposed/train_proposed.py` | ByT5 standard baseline trainer |
| `models/proposed/noise_robust_model.py` | Earlier ByT5 architecture (superseded) |
| `notebooks/phase2/byt5_kaggle_all_levels.ipynb` | Kaggle: ByT5 standard |
| `notebooks/phase2/gpu_baselines_colab.ipynb` | Kaggle/Colab: mBERT/XLM-R/MuRIL |
| `notebooks/phase2/charcnn_medium_high.ipynb` | Friend's notebook: CharCNN med+high |
| `data/final/` | All 7 split CSVs (train/val/test x4 noise) |
| `results/tables/cpu_results/` | TF-IDF SVM/LR + CharCNN results |
| `results/tables/Gpu_results/` | mBERT + GPU model results |
