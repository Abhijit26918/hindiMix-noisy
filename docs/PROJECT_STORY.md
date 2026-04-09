# Project Story: How We Got Here

A complete record of decisions, failures, pivots, and the reasoning behind them.
Use this to explain the project to your advisor, committee, or anyone who asks.

---

## The Original Idea

Started with a simple question: hate speech detection systems are trained on clean text,
but real-world speech-to-text (ASR) output is noisy — especially for Hindi-English
(Hinglish) which ASR systems handle poorly. Does this hurt performance? Can we fix it?

Target: build a dataset, run experiments, propose a fix, publish at EMNLP/COLING 2026.

---

## Phase 1: Data — What We Tried and What Failed

### Attempt 1: Download everything automatically
Tried to download 6 datasets using HuggingFace `datasets` library.

**Failure:** `trust_remote_code=True` deprecated in newer HuggingFace. Many datasets
that use custom loading scripts threw errors and couldn't load.

**Fix:** Changed to `trust_remote_code=False`, switched to direct file downloads
for datasets that failed. Used `requests` + GitHub raw URLs instead of HF API.

---

### Attempt 2: Hindi-English data from HuggingFace
Checked all HuggingFace datasets for Hindi-English code-mixed hate speech.

**Problem:** Nothing suitable. The SemEval 2020 Task 9 Sentimix dataset exists but
was not auto-downloadable — the official link required Google Drive access.

**Fix:** Found the data in a GitHub repo (singhnivedita/SemEval2020-Task9).
Downloaded directly from raw GitHub URLs. Got 18,631 Hinglish tweets.

**Catch:** All SemEval labels are sentiment (0/1/2), not hate speech.
So these 18,631 rows are Hinglish text but label=0 for all.
Useful as code-mixed text source but not as labeled hate speech.

---

### Attempt 3: HASOC from HuggingFace
HASOC 2021 CodeMix was supposed to have labeled Hindi-English hate speech.
HuggingFace version failed to load.

**Fix:** User downloaded the raw dataset manually from GitHub
(AditiBagora/Hasoc2021CodeMix). The data is in nested JSON format:
- Per topic folder (covid, indian politics, charlie hebdo, etc.)
- Per tweet thread: data.json (tweet text + replies) + labels.json (HOF/NONE)

Wrote `parse_hasoc_json.py` to recursively extract tweets + join with labels.

**Result:** 2,819 tweets parsed. HOF=1,309, NONE=1,510. This is the real
labeled Hinglish hate speech data that makes the paper legitimate.

---

### Attempt 4: Fixing the data pipeline (codemixed_clean.csv = 0 rows)
After merging all datasets, `codemixed_clean.csv` had 18,631 rows but ALL label=0.

**Root cause:** Only SemEval was being recognized as hi-en. HASOC wasn't merged yet.
Also, script 04 was using codemixed_clean.csv for splits — giving all-zero labels.

**Fix:** Script 04 changed to use merged_clean.csv (all sources, balanced labels).
After HASOC integration: 21,450 hi-en samples (HASOC 2,819 + SemEval 18,631).

---

### Attempt 5: Index alignment bug in split creation
Noisy test sets were built by matching indices between clean and noisy dataframes.
After filtering, indices misaligned — wrong rows matched together.

**Fix:** Match on `text_original` column (the original text before noise injection)
instead of integer indices. Stable across filtering operations.

---

## Phase 2: Baselines — What Ran, What Failed

### CPU baselines (TF-IDF + CharCNN)
Ran in Colab CPU notebook. Completed successfully.
Results in `results/tables/cpu_results/`.

**Note:** These results are valid — no GPU needed, fast to reproduce.

---

### GPU baselines: Colab vs Kaggle
Initially planned to use Google Colab GPU.

**Problem:** Colab GPU sessions disconnect. Training 4 noise levels x 3 models = 12 runs
takes hours. Colab disconnects before finishing.

**Fix:** Switched to Kaggle which gives 30 hrs/week free GPU and doesn't disconnect.
User adapted the Colab notebook to Kaggle manually.

**Problem 2:** User has multiple models to train simultaneously.
**Fix:** Used multiple Kaggle accounts to parallelize.

---

### ByT5 OOM on Kaggle (T4 GPU, 15GB VRAM)
First run crashed immediately with CUDA out of memory.

**Root cause:** ByT5-small operates at byte level — sequences are ~4x longer than
subword tokenized sequences. With batch_size=16 and MAX_LEN=256, activation memory
exceeded 15GB.

**Fix (3 changes):**
1. batch_size: 16 → 4
2. MAX_LEN: 256 → 128 (128 bytes ≈ 100 characters, enough for tweets)
3. `model.encoder.gradient_checkpointing_enable()` — recomputes activations
   instead of storing them, saves ~30% VRAM

---

### ByT5 device error (CPU tensor vs GPU model)
`RuntimeError: Expected all tensors to be on the same device`

**Root cause:** `PhoneticFeatureExtractor.compute_phonetic_features()` creates tensors
using `torch.tensor(...)` which defaults to CPU. The linear layer (`self.proj`) is
on GPU after `.to(DEVICE)`.

**Fix:** Added `feats = feats.to(self.proj.weight.device)` in the forward method.
Also: padding zeros created with `device=hidden.device` to avoid same issue.

---

### GradScaler deprecation warning
`torch.cuda.amp.GradScaler` deprecated in newer PyTorch.
**Fix:** Changed to `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`.

---

## Phase 3: Proposed Model — Design Decisions

### Version 1: ByT5 + phonetic features + noise-aware attention
Built `noise_robust_model.py`. Simple: ByT5 encoder + phonetic projection + sigmoid gate.

**Weakness identified:** Not mathematically novel. ByT5 exists. Phonetic features exist.
Sigmoid gate is trivial. Not publishable at EMNLP level.

---

### Version 2: Adding contrastive learning (PWNIC)
**Insight:** We already have clean/noisy pairs in train.csv. Contrastive learning
needs exactly this — positive pairs of (clean, noisy) versions of the same text.

Standard approach (SimCSE): push (clean, noisy) embeddings close in output space.
**Novel twist:** Weight the contrastive pressure by phonetic dissimilarity.

`phi = 1 - JaroWinkler(clean_text, noisy_text)`

If "badmaash" → "bdms" (very different phonetically), phi is high → push harder.
If "badmaash" → "badmaas" (very similar), phi is low → gentle push.

Standard InfoNCE treats all pairs equally. PWNIC does not. This is the mathematical novelty.

---

### Version 3: Adding Gradient Reversal Layer
**Problem with PWNIC alone:** It aligns representations in output space but doesn't
prevent noise information from remaining in the feature representation. The classifier
could still use noise-correlated features to make decisions (fragile to new noise patterns).

**Solution:** Gradient Reversal Layer (DANN, Ganin et al. 2016) applied to noise level prediction.

Standard use: domain adaptation (source → target domain).
Our use: noise robustness (clean → noisy text).

The GRL forces the encoder into a minimax game:
- Noise predictor: minimize L_adv (correctly predict noise level from encoder output)
- Encoder: maximize L_adv via reversed gradients (make representation noise-level-unpredictable)

Result: encoder must preserve hate semantics (needed for L_CE) while removing noise
characteristics (forced by -L_adv). Representation disentanglement at feature level.

**DANN annealing:** Lambda starts at 0, rises to 1.0 via schedule lambda(p) = 2/(1+exp(-10p))-1.
Prevents adversarial signal from destabilizing early training when encoder hasn't
learned hate semantics yet.

**Why this is novel:** GRL has been used for domain adaptation, language, sentiment.
Never applied specifically to ASR noise level disentanglement in code-mixed hate speech.

---

### Final Architecture: NoiseBridge
```
L_total = L_CE + 0.5*L_PWNIC - 0.3*L_adv + 0.1*L_aux
```

4 loss terms, each attacking the problem differently:
- L_CE: learn to detect hate
- L_PWNIC: output space — align clean/noisy embeddings with phonetic weighting
- L_adv (with GRL): feature space — remove noise info from representation
- L_aux: multi-task — separate branch learns noise awareness positively

The key insight: PWNIC and GRL attack noise robustness from opposite directions.
PWNIC says "clean and noisy should look similar in output space."
GRL says "the encoder should not know what noise level it's seeing."
Together they are stronger than either alone — that tension is the contribution.

---

## Challenges Summary

| Challenge | Root Cause | Fix |
|---|---|---|
| HuggingFace datasets failing | trust_remote_code deprecated | Direct file downloads |
| No labeled Hinglish hate data | HASOC not auto-downloadable | Manual download + parse_hasoc_json.py |
| codemixed splits all label=0 | Wrong CSV used for splits | Use merged_clean.csv not codemixed_clean.csv |
| Index alignment bug | Integer index fragile after filter | Match on text_original column |
| Colab GPU disconnecting | Long training sessions | Switched to Kaggle |
| ByT5 OOM | Byte sequences 4x longer, large hidden size | batch=4, len=128, gradient checkpointing |
| CPU/GPU tensor mismatch | Phonetic tensors created on CPU | .to(self.proj.weight.device) |
| Architecture not novel enough | First design too simple | PWNIC + GRL = real contribution |

---

## Expectations vs Reality

| Expectation | Reality |
|---|---|
| HuggingFace datasets "just work" | Many need manual download/parsing |
| SemEval = hate speech labels | SemEval = sentiment labels (0/1/2), not hate |
| HASOC on HuggingFace works | Had to parse nested JSON manually |
| ByT5 fits on T4 GPU easily | OOM — needed 3 fixes to fit |
| Simple architecture is enough | Needed PWNIC + GRL for real novelty |
| All data is code-mixed | 81% is English, only 19% is Hinglish |

---

## What Makes This Paper Defensible

1. **First benchmark:** No prior work evaluates hate speech detection under ASR-style
   noise at multiple noise levels on Hinglish. We created this benchmark.

2. **Novel loss function:** PWNIC — phonetically-weighted contrastive loss. The weighting
   by JaroWinkler dissimilarity is not in any prior contrastive NLP paper.

3. **Novel application of GRL:** Applying gradient reversal to noise-level disentanglement
   (not domain adaptation) for code-mixed hate speech is new.

4. **Motivated design:** Every component traces back to a specific weakness of existing
   approaches. Not "we tried things and this worked" but "we identified root causes
   (subword OOV, noise-correlated features) and directly addressed them."

5. **Honest scope:** If NoiseBridge underperforms on clean data but has lower degradation,
   that's still the paper's point — robustness, not absolute performance.
