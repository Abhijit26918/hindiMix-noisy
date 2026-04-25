# Project Story: How We Got Here
# NoiseBridge — Complete Journey for BTP-1 Report

A full record of every decision, failure, pivot, bug, and result — from blank slate to final paper.
Written for: BTP-1 submission, April 25, 2026.

---

## The Original Idea

Started with a simple question: hate speech detection systems are trained on clean typed text,
but real-world content often arrives through ASR (Automatic Speech Recognition) pipelines
as noisy transcripts — especially for Hindi-English (Hinglish) where ASR error rates are 15–40%.
Does this hurt performance? Can we fix it?

No prior work had studied this specific intersection: hate speech + code-mixed text + ASR noise.

**Target:** Build a benchmark dataset, run experiments, propose a novel model, publish at EMNLP/COLING 2026.

---

## Phase 1: Data Collection — What We Tried and What Failed

### Goal
Aggregate labeled hate speech data with Hinglish content, then inject controlled ASR-style noise
to create a multi-level noisy benchmark.

### Attempt 1: HuggingFace auto-download
Tried to download 6 datasets using HuggingFace `datasets` library.

**Failure:** `trust_remote_code=True` was deprecated in newer HuggingFace. Many datasets
using custom loading scripts threw errors.

**Fix:** Switched to `trust_remote_code=False`. For datasets that failed, used direct
`requests` + GitHub raw URL downloads instead of the HF API.

---

### Attempt 2: Getting Hindi-English data
Searched all HuggingFace datasets for Hindi-English code-mixed hate speech.

**Problem:** SemEval 2020 Task 9 (Sentimix) was not auto-downloadable — required Google Drive.
Also, critically: SemEval labels are *sentiment* (0/1/2), NOT hate speech labels.

**Fix:** Downloaded from `singhnivedita/SemEval2020-Task9` on GitHub.
Got 18,631 Hinglish tweets — used as code-mixed *text source* only, all assigned label=0 (non-hate).

**Lesson:** Always verify label schema before assuming a dataset is directly usable.

---

### Attempt 3: HASOC 2021 — the real Hinglish hate speech
HASOC 2021 CodeMix was the actual labeled Hindi-English hate speech dataset we needed.
HuggingFace version failed to load.

**Fix:** Manually downloaded from `AditiBagora/Hasoc2021CodeMix` on GitHub.
Data was in nested JSON format: per-topic folders (COVID, Indian politics, etc.), each with
`data.json` (tweet text + replies) and `labels.json` (HOF/NONE labels).

Wrote `parse_hasoc_json.py` to recursively extract tweets and join with labels.

**Result:** 2,819 labeled tweets. HOF=1,309, NONE=1,510.
This is the core of what makes the paper legitimate — real, labeled Hinglish hate speech.

---

### Attempt 4: Label pipeline bug — codemixed_clean.csv all zeros
After merging all datasets, `codemixed_clean.csv` had 18,631 rows but ALL label=0.

**Root cause:** Only SemEval was being recognized as hi-en. HASOC wasn't merged yet.
Script 04 was also using `codemixed_clean.csv` for splits — giving all-zero labels throughout.

**Fix:** Changed script 04 to use `merged_clean.csv` (all sources, proper labels).
After HASOC integration: 21,450 hi-en samples in the dataset.

---

### Attempt 5: Index alignment bug
Noisy test sets were built by matching rows between clean and noisy dataframes.
After filtering, integer indices misaligned — wrong rows were matched together.

**Fix:** Match on the `text_original` column (the source text before noise injection)
instead of integer indices. Stable across any filtering operation.

---

### Final Dataset: HindiMix-Noisy Benchmark

| Source | Examples | Notes |
|--------|----------|-------|
| Davidson et al. (2017) | 17,290 | English tweets |
| OLID | 9,774 | English offensive |
| SemEval 2020 Task 9 | 18,631 | Hindi-English (text only, label=0) |
| HASOC 2021 | 7,952 | Hindi-English labeled hate/non-hate |
| UCB Measuring Hate | 27,854 | English |
| TweetEval Hate | 8,909 | English |
| Web scraped | 34,293 | Mixed |
| **Total** | **123,631** | Before cleaning |

**Noise injection** (per character, 4 operations at prob p/4 each):
1. Substitution: replace with random alphabet char
2. Insertion: keep char, insert random char after
3. Deletion: remove char
4. Word split: insert space after char (simulates ASR word boundary errors)

Perturbation probabilities: low=0.05, medium=0.10, high=0.20 — calibrated against
empirically measured Hinglish ASR error rates of 15–40%.

---

## Phase 2: Data Cleaning — Discovered Later

### Problem found (Apr ~15, 2026)
When reviewing the data, discovered ~6.5% of training rows were not real text —
they were corrupted tweet IDs and spreadsheet artifacts from HASOC 2021.

### Root cause
HASOC 2021 tweet IDs are 18-digit integers. Pandas reads them as `float64` when any NaN
exists in the column, losing precision (e.g., `1385642367265370112` → `1385642367265370000`).
The noise injection pipeline then transformed these IDs further — inserting spaces, substituting
chars, converting to scientific notation via Excel — creating a whole family of junk variants.

### 9 junk categories removed (in order applied)

| # | Pattern | Example | Cause |
|---|---------|---------|-------|
| 1 | Pure numeric ≥15 digits | `1392749481536286721` | Raw tweet ID |
| 2 | Digits + whitespace only | `13971018 27116703744` | Tweet ID + injected space |
| 3 | One alpha + rest numeric | `13s0940892903608321` | Tweet ID + one char substituted |
| 4 | Scientific notation | `1.39E+21` | Excel float64 overflow of tweet ID |
| 5 | `#NAME?` | `#NAME?` | Excel formula error artifact |
| 6 | Tweet ID + filler word | `1392749481536286721 umm` | ID + short noise word |
| 7 | Zero-width spaces (`\u200b`) | real text + `\u200b` | Stripped only, row KEPT |
| 8 | Arrow + letter artifacts | `v   ⇒`, `b   ⇒` | Noise corruption of short alpha (≤3) |
| 9 | NaN artifacts | `_… nan` | NaN serialization artifact |
| 10 | Noise-corrupted name fragments | `yadav88`, `a77 pak` | Usernames/number fragments |

### Final row counts after cleaning

| File | Original | After cleaning | Removed |
|------|----------|----------------|---------|
| train.csv | 123,631 | 115,554 | ~8,077 (~6.5%) |
| val.csv | 16,882 | 16,463 | 419 |
| test_clean.csv | 16,882 | 16,431 | 451 |
| test_noisy_low.csv | 16,882 | 16,431 | 451 |
| test_noisy_medium.csv | 16,882 | 16,449 | 433 |
| test_noisy_high.csv | 16,882 | 16,428 | 454 |

**Why this matters for the paper:** The cleaning required 9 targeted passes rather than a
single filter — each pass exposed a new artifact type produced by the noise pipeline on the
corrupt inputs. This is worth one paragraph in the Dataset Preprocessing section.

---

## Phase 2: Baselines — CPU and GPU

### CPU baselines (TF-IDF + CharCNN)
Ran in Colab/local notebook. Completed successfully.
Results in `results/tables/cpu_results/`.

### GPU baselines: Colab → Kaggle switch
Initially planned Google Colab GPU.

**Problem:** Colab disconnects after a few hours. Training 4 noise levels × 3 models = 12 runs,
each taking hours. Sessions timed out before finishing.

**Fix:** Switched to Kaggle — 30hrs/week free GPU, persistent sessions, no disconnections.
Used multiple Kaggle accounts to run mBERT, XLM-R, MuRIL in parallel.

### ByT5 OOM on Kaggle T4 (15GB VRAM)
First ByT5 run crashed immediately — CUDA out of memory.

**Root cause:** ByT5 operates at byte level — sequences are ~4× longer than subword-tokenized
sequences. With batch_size=16 and MAX_LEN=256, activation memory exceeded 15GB.

**Fix (3 changes):**
1. batch_size: 16 → 4
2. MAX_LEN: 256 → 128
3. `model.encoder.gradient_checkpointing_enable()` — recomputes activations, saves ~30% VRAM

### ByT5 baseline broken — T5ForSequenceClassification wrong
Early ByT5 results were 0.27 F1 — barely above random.

**Root cause:** Used `T5ForSequenceClassification` which includes both encoder AND decoder.
For classification, only the encoder is needed. The decoder adds noise and massively inflates memory.

**Fix:** Changed to `T5EncoderModel` + masked mean pooling + weighted cross-entropy.
Results improved substantially.

**Final decision on ByT5:** Dropped from final paper. Performance similar to MuRIL but
at much higher compute cost and complexity. Not worth the trade-off for the paper's claims.

### Final Baseline Results

| Model | Clean | Low | Medium | High |
|-------|-------|-----|--------|------|
| mBERT | 0.8428 | 0.8390 | 0.8305 | 0.8267 |
| XLM-R | 0.8467 | 0.8417 | 0.8396 | 0.8294 |
| MuRIL | 0.8389 | 0.8365 | 0.8312 | 0.8250 |

All transformers show degradation clean→high of ~1.4–1.7 F1 points. Lexical baselines degrade
more steeply (~2.7 points), confirming transformers have built-in noise robustness via
contextual representations.

---

## Phase 3: Proposed Model — Architecture Evolution

### Version 1: ByT5 + phonetic features + sigmoid gate
Built `noise_robust_model.py`. ByT5 encoder + phonetic feature projection + sigmoid gate
to suppress noisy tokens.

**Weakness:** Not mathematically novel. ByT5 is existing work. Phonetic features are existing work.
Sigmoid gate is trivial. Not defensible at EMNLP level.

---

### Version 2: Adding PWNIC Loss

**Insight:** The training set already contains paired (clean, noisy) versions of the same text.
Contrastive learning needs exactly this. Standard approach (SimCSE) pushes (clean, noisy)
embeddings together — but treats all pairs identically.

**Novel twist:** Weight the contrastive pressure by how phonetically different the pair is.

```
φ_i = 1 - (1/M) Σ_j JaroWinkler(word_j_clean, word_j_noisy)
L_PWNIC = -(1/Σφ) · Σ_i φ_i · log[exp(sim(p_i^c, p_i^n)/τ) / Σ_{k≠i} exp(sim(p_i^c, p_k)/τ)]
```

If "badmaash" → "bdms" (very different phonetically): φ is high → push harder.
If "badmaash" → "badmaas" (barely changed): φ is low → gentle push.

Standard InfoNCE treats all pairs equally. PWNIC does not. This is the mathematical novelty.

**Why Jaro-Winkler:** Bounded [0,1], gives credit for transpositions and prefix matches,
which is phonetically motivated. Edit distance is unbounded and treats all errors equally.

---

### Version 3: Adding Gradient Reversal Layer (GRL)

**Motivation:** PWNIC aligns output representations but doesn't remove noise information
from the feature space. The classifier could still exploit noise-correlated features.

**Solution tried:** Gradient Reversal Layer (Ganin et al. 2016) applied as a noise-level
adversarial predictor. Encoder is forced into a minimax game — the adversarial head tries
to predict noise level, the GRL reverses the gradient to make noise level *unpredictable*.

Architecture v3 loss:
```
L_total = L_CE + 0.5*L_PWNIC + 0.3*L_adv (via GRL) + 0.1*L_aux
```

**This failed catastrophically:**
- XLM-R high noise with GRL: **0.777** (−5.2 pts from baseline 0.829)
- MuRIL medium noise with GRL: **0.786** (−4.5 pts from baseline 0.831)

**Why it failed — two root causes:**
1. **Degenerate targets:** When a batch contains only one noise level (which happens frequently),
   the adversarial predictor trivially converges — it just always predicts that level. The reversed
   gradient then pushes the encoder toward uniform, content-free representations, destroying
   classification ability.
2. **Contradictory gradients:** The auxiliary head (γ=0.1) tries to make noise level detectable.
   The GRL (β=0.3) tries to make it undetectable. Both operate on the same `z^n`. Since β > γ,
   GRL wins — and swamps the aux signal entirely.

**Contrast with DANN (where GRL works):** Domain adaptation always has structurally distinct domains
in every batch. The degenerate target failure mode is absent. This structural guarantee doesn't hold
for noise levels in our training setup.

**Decision:** GRL removed. The failure is documented as the third empirical contribution — it answers
the question "why doesn't adversarial disentanglement work here?" and provides genuine insight for
future work on noise robustness.

---

### Final Architecture: NoiseBridge (v2, no GRL)

```
L_total = L_CE + 0.15 * L_PWNIC + 0.20 * L_aux
```

**Components:**
1. **Multilingual encoder** (mBERT / MuRIL / XLM-R backbone)
2. **Noise-Aware Attention gate:** `z = mean_pool(H ⊙ σ(W_g · H))`
   — learned per-token scalar gate; suppresses corrupted tokens when combined with PWNIC signal
3. **PWNIC loss** (novel) — phonetically-weighted contrastive objective via projection head
4. **Auxiliary noise-level classifier** — regularises noisy representation to be noise-aware;
   auto-disabled if batch has <2 distinct noise levels (prevents degenerate training)

**Why gate alone hurts (−0.55 F1):** The gate has no signal for what "corrupted" means without
PWNIC. It suppresses arbitrary tokens. PWNIC provides the signal: corrupted tokens should be
alignable with their clean counterparts, so the gate learns to de-emphasise tokens where
alignment is hardest. Gate + PWNIC are co-dependent by design.

---

## Phase 4: Training Issues and Fixes

### Early NoiseBridge results were worse than baselines
All three NoiseBridge models lost to their baselines before the Opus 4.7 session.

| Problem | Fix |
|---------|-----|
| No class weights in CE loss | Added `class_weights=[1.66, 1.0]` (hate is minority at 37.6%) |
| α=0.5 too high — PWNIC dominated CE | Lowered to α=0.15 |
| PyTorch 2.1 AMP API mismatch | Changed `torch.cuda.amp.GradScaler()` → `torch.amp.GradScaler('cuda')` |
| GRL causing collapse | Removed GRL entirely from final model |
| Aux head always-on with single noise level batches | Auto-disable when `len(set(noise_labels)) < 2` |

### GradScaler API (PyTorch 2.1)
`torch.cuda.amp.GradScaler` is deprecated in PyTorch 2.1.
**Fix:** `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`.

**Note for P100 specifically:** P100 (sm_60) does not support BF16 — FP16 only.
Use `torch.amp.autocast('cuda', dtype=torch.float16)` explicitly.

---

## Phase 5: Final Results

All experiments: 3 encoders × 3 seeds (42, 43, 44) × 4 noise conditions = 36 runs.
Results in `results/tables/final_results/` as individual JSON files.

### NoiseBridge vs Baseline (Macro-F1, mean ± std over 3 seeds)

| Model | Clean | Low | Medium | High | Avg |
|-------|-------|-----|--------|------|-----|
| **XLM-R baseline** | 0.8467 | 0.8417 | 0.8396 | 0.8294 | 0.8394 |
| **XLM-R + NoiseBridge** | **0.8498 ± .002** | **0.8453 ± .003** | **0.8435 ± .002** | **0.8350 ± .002** | **0.8434** |
| mBERT baseline | 0.8428 | 0.8390 | 0.8305 | 0.8267 | 0.8347 |
| mBERT + NoiseBridge | 0.8424 ± .003 | 0.8385 ± .004 | 0.8320 ± .004 | 0.8273 ± .004 | 0.8350 |
| MuRIL baseline | 0.8389 | 0.8365 | 0.8312 | 0.8250 | 0.8329 |
| MuRIL + NoiseBridge | 0.8401 ± .002 | 0.8358 ± .003 | 0.8305 ± .001 | 0.8237 ± .002 | 0.8325 |

**XLM-R gains:** +0.31 clean, +0.36 low, +0.39 medium, +0.56 high (largest at highest noise — expected)
**mBERT / MuRIL:** essentially neutral — NoiseBridge doesn't hurt them, just doesn't help

### Why the tokeniser determines everything
XLM-R uses **SentencePiece** — operates on raw Unicode, preserving character-level noise traces.
mBERT / MuRIL use **WordPiece** — applies case folding and normalisation before tokenising,
erasing exactly the phonetic variations PWNIC relies on.
By the time WordPiece models see the input, the signal PWNIC needs is already gone.

### Component Ablation (XLM-R, seed 42, avg F1)

| Config | Avg F1 |
|--------|--------|
| Baseline CE only | 0.8394 |
| + Gate only | 0.8339 (−0.55, hurts) |
| + Gate + PWNIC | 0.8401 |
| + Gate + PWNIC + Aux | **0.8434** |
| + Gate + PWNIC + Aux + GRL | 0.8237 (−1.57, catastrophic) |

---

## Phase 6: Paper + Documentation (Opus 4.7 Session)

After experiments completed, went to Claude Opus 4.7 (web) for a deep paper revision session.

**What was produced:**
- `docs/Noisebridge_v5.pdf` — final IEEE-format paper, double-column, 2 authors
- `docs/RESEARCH_DEEP_REFERENCE.md` — 630-line panel survival guide with:
  - Full math derivations for all loss functions
  - Complete Q&A for every expected panel question
  - Board-solvable worked examples (φ calculation, PWNIC→InfoNCE reduction, macro-F1)
  - Annotated bibliography (11 must-know papers with what to say about each)
  - Quick-reference cheat sheet with all key numbers

**Key paper decisions from that session:**
- GRL failure reframed as third contribution (not a failure of the paper)
- SpokenCSE (Chang & Chen, Interspeech 2022) added as key related work to differentiate from
- "Why XLM-R but not others" explanation (SentencePiece vs WordPiece) sharpened
- Bibliography ordered by first appearance (IEEE requirement), 27 entries
- De-AI'd writing in Discussion / Future Work / Conclusion sections

---

## Summary: What Makes This Paper Defensible

1. **First benchmark:** HindiMix-Noisy is the first dataset with controlled multi-level ASR noise
   for code-mixed (Hinglish) hate speech. 115,554 examples across 4 noise conditions.

2. **Novel loss function:** PWNIC — phonetically-weighted contrastive loss. Weighting InfoNCE by
   Jaro-Winkler phonetic dissimilarity is not in any prior contrastive NLP paper.

3. **Honest negative result:** GRL was theoretically motivated, implemented correctly, and failed
   catastrophically. The failure is explained (degenerate targets + contradictory gradients) and
   documented — this is more valuable than hiding it.

4. **Motivated design:** Every component traces to a specific identified weakness:
   - Gate → suppresses corrupted tokens
   - PWNIC → gives the gate a signal to learn from
   - Aux → regularises noise-aware representation
   - No GRL → degenerate targets would collapse the encoder

5. **Honest limitations:** Synthetic noise (not real ASR), English-heavy data (51.6% English),
   modest gains (+0.41 avg F1 — headroom limited by transformers' inherent robustness),
   zero improvement on WordPiece encoders.

---

## Challenges Summary

| Challenge | Root cause | Fix |
|-----------|-----------|-----|
| HuggingFace datasets failing | trust_remote_code deprecated | Direct GitHub raw URL downloads |
| No labeled Hinglish hate data | HASOC not auto-downloadable | Manual download + parse_hasoc_json.py |
| SemEval has sentiment not hate labels | Dataset mislabeled in planning | Used SemEval as text-only source, label=0 |
| codemixed splits all label=0 | Wrong CSV used for splits | Use merged_clean.csv not codemixed_clean.csv |
| Index alignment bug | Integer index fragile after filter | Match on text_original column |
| Colab GPU disconnecting | Long training sessions | Switched to Kaggle |
| ByT5 OOM on T4 | Byte sequences 4× longer | batch=4, len=128, gradient checkpointing |
| ByT5 baseline F1=0.27 | T5ForSequenceClassification wrong | T5EncoderModel + masked mean pool |
| ByT5 device error | Phonetic tensors created on CPU | .to(self.proj.weight.device) |
| GradScaler API changed | PyTorch 2.1 deprecation | torch.amp.GradScaler('cuda') |
| NoiseBridge losing to baselines | No class weights, α too high, AMP bug | Fixed all three, re-ran all seeds |
| GRL catastrophic failure | Degenerate targets + contradictory grads | Removed GRL, documented failure |
| 6.5% of training data was junk | HASOC tweet IDs as float64 + noise pipeline | 9-pass cleaning on data/final/ CSVs |
| Data duplicated in final_results/ | JSON files in both root and subfolders | Deduplicate by (encoder, seed, test_noise) key |

---

## Expectations vs Reality

| Expectation | Reality |
|-------------|---------|
| HuggingFace datasets "just work" | Many need manual download/parsing |
| SemEval = hate speech labels | SemEval = sentiment labels (0/1/2), not hate |
| HASOC on HuggingFace works | Had to parse nested JSON manually |
| ByT5 fits on T4 GPU easily | OOM — needed 3 fixes to fit |
| ByT5 baseline will be strong | F1=0.27 due to wrong model class; after fix, similar to MuRIL |
| GRL will disentangle noise | Catastrophic collapse — degenerate targets |
| All architectures benefit from PWNIC | Only XLM-R benefits; WordPiece destroys the signal |
| Simple architecture is enough | Needed PWNIC + ablation + negative result for novelty |
| All data is code-mixed | 81% is English, only 19% is Hinglish |
| Results would be clean (no junk rows) | 6.5% of train was tweet IDs / artifacts |

---

## File Map (Key Files)

| File | Purpose |
|------|---------|
| `docs/Noisebridge_v5.pdf` | Final paper draft |
| `docs/icaiac-paper-file/noisebridge_revised.tex` | LaTeX source |
| `docs/RESEARCH_DEEP_REFERENCE.md` | Panel prep + full math + Q&A |
| `docs/PROJECT_STORY.md` | This file — full narrative for BTP report |
| `docs/REFERENCES.md` | All 30+ citations annotated |
| `models/proposed/noisebridge_new.py` | NoiseBridge model (final, no GRL) |
| `models/proposed/train_noisebridge_new.py` | Training script (final) |
| `data/final/` | All cleaned CSVs (train/val/test × 4 conditions) |
| `results/tables/final_results/` | All NoiseBridge JSONs (3 models × 3 seeds × 4 conditions) |
| `results/tables/Gpu_results/` | Baseline transformer results |
| `results/tables/cpu_results/` | TF-IDF SVM/LR + CharCNN results |
| `results/figures/` | All paper figures (PNG) |
| `docs/icaiac-paper-file/` | ICAIAC submission files + extra figures |
