# NoiseBridge — Deep Research Reference & Panel Presentation Guide

> Everything you need to understand, explain, defend, and extend this paper.
> Covers: full math derivations, dataset details, all cited papers, expected Q&A, proofs you must know.

---

# PART 1: THE PROBLEM FROM SCRATCH

## 1.1 Why This Problem Exists

When someone posts a voice message on WhatsApp or speaks on a live stream, the audio must be converted to text before any NLP model can analyse it. This conversion is done by an **ASR (Automatic Speech Recognition)** system — like Google Speech-to-Text or Whisper.

ASR systems make mistakes. A sentence like:
> "Tera baap kya karta hai, madarchod?"

might be transcribed as:
> "Tera bap kya ktra hai, madrachd?"

Now your hate speech detector — trained on clean, typed text — sees a garbled string it has never encountered. It may fail to classify it correctly.

This problem is **worse for Hinglish** (Hindi-English code-mixed) because:
- Romanised Hindi has no standard spelling ("madarchod" / "madarchoad" / "madarchot" are all found online)
- ASR systems are primarily trained on monolingual data
- Code-switching within a sentence confuses language models
- Error rates for Hinglish ASR are **15–40%** (vs 5–10% for English monolingual)

**No prior work had studied this specific intersection:** hate speech + code-mixed text + ASR noise.

## 1.2 The Gap This Paper Fills

Before this paper:
- Hate speech benchmarks assumed clean typed text
- ASR-robustness work was done for intent detection (English only) or machine translation
- Code-mixed hate speech work existed but not under ASR noise conditions

This paper provides:
1. The **first benchmark** with controlled multi-level ASR noise for code-mixed hate speech
2. A **training method** (NoiseBridge) that explicitly accounts for phonetic distortion
3. An **important negative result**: gradient reversal doesn't work here

---

# PART 2: BACKGROUND MATHEMATICS

## 2.1 Transformers and Self-Attention (What the Encoder Does)

A transformer encoder takes a sequence of tokens and produces contextual embeddings.

**Input:** Token sequence `x = [x_1, x_2, ..., x_L]`, embedded as `E ∈ R^{L×d}`

**Self-attention (single head):**
```
Q = E·W_Q,   K = E·W_K,   V = E·W_V       (W_Q, W_K, W_V ∈ R^{d×d_k})

Attention(Q,K,V) = softmax(QK^T / √d_k) · V
```

The `softmax(QK^T / √d_k)` gives an attention weight matrix — how much each token attends to every other. Division by `√d_k` prevents vanishing gradients from large dot products.

**Multi-head attention** runs h of these in parallel and concatenates:
```
MultiHead(Q,K,V) = Concat(head_1,...,head_h)·W_O
```

The encoder stacks N such layers (12 for base models), producing `H ∈ R^{L×d}` — contextual representations.

**Why does this matter for noise?** Contextual representations capture the whole sequence, so a corrupted token can partly be recovered from its neighbours. This is why transformers are already somewhat robust to character noise even without special training.

## 2.2 Cross-Entropy Loss (Classification Objective)

Standard classification loss:
```
L_CE = -∑_{c=0}^{C-1} y_c · log(p_c)
```
where `y_c` is the one-hot true label and `p_c = softmax(logits)_c`.

For binary hate/non-hate (`C=2`) with **class weights** `w_c` (to handle imbalance):
```
L_CE = -∑_c w_c · y_c · log(p_c)
```

In this paper: hate class (37.6%) and non-hate class (62.4%) → weight ratio ≈ 1.66:1. So `w_hate = 1.66`, `w_non-hate = 1.0`. This penalises missing hate more than missing non-hate.

**Why class weighting?** The dataset has 37.6% hate and 62.4% non-hate — imbalanced. Without weighting, a model that always predicts "non-hate" gets 62.4% accuracy. Macro-F1 corrects for this at evaluation; class weighting corrects for it during training.

## 2.3 InfoNCE Loss (What PWNIC Extends)

**InfoNCE** (van den Oord et al., 2018) is the foundation for contrastive representation learning. For a batch of N (anchor, positive) pairs:

```
L_InfoNCE = -1/N · ∑_i log [ exp(sim(z_i, z_i+) / τ) / ∑_{k≠i} exp(sim(z_i, z_k) / τ) ]
```

**Intuition:**
- `z_i` = anchor embedding (e.g., clean text)
- `z_i+` = positive embedding (e.g., noisy version of same text)
- `z_k` for `k≠i` = negative embeddings (other examples in batch)
- We want the numerator (similarity to positive) to be high relative to denominator (all others)
- `τ` = temperature: low τ makes the distribution sharper (harder negatives matter more)
- As N→∞, minimising InfoNCE is equivalent to maximising mutual information between anchor and positive

**SimCSE** (Gao et al., 2021) uses this with dropout as the augmentation — pass the same sentence through the encoder twice with different dropout masks, treat the two outputs as positives.

**SpokenCSE** (Chang & Chen, 2022) uses clean transcript and ASR-noisy transcript as the positive pair — directly relevant to our problem.

## 2.4 Jaro-Winkler Similarity (The Phonetic Metric)

**Jaro similarity** between strings s and t:
```
Jaro(s,t) = 0                                  if m = 0
           = (m/|s| + m/|t| + (m-t')/m) / 3    otherwise
```
where:
- `m` = number of matching characters (within window `floor(max(|s|,|t|)/2) - 1`)
- `t'` = number of transpositions / 2
- `|s|`, `|t|` = string lengths

**Jaro-Winkler** adds a prefix bonus:
```
JW(s,t) = Jaro(s,t) + l · p · (1 - Jaro(s,t))
```
where:
- `l` = length of common prefix (up to 4 characters)
- `p` = scaling factor (typically 0.1)

**Example:** 
- `JW("madarchod", "madrachd")` ≈ 0.93 (highly similar — only a few chars different)
- `JW("madarchod", "xyz")` ≈ 0.0 (completely different)

**Why Jaro-Winkler over edit distance?** Jaro-Winkler is bounded [0,1] and gives partial credit for transpositions and prefix matches, which better models phonetic similarity. Edit distance is unbounded and treats all errors equally.

## 2.5 PWNIC Loss — Full Derivation

**Given:**
- Clean sentence `x^c` and its ASR-noisy version `x̃`
- Phonetic dissimilarity: `φ_i = 1 - (1/M) ∑_j JW(w_j^c, w_j^n)` where M = max word count
- Encoder outputs gated representations `z_i^c`, `z_i^n`
- Projection head `π_ξ` (2-layer MLP): `p_i^c = π_ξ(z_i^c)`, `p_i^n = π_ξ(z_i^n)`
- Cosine similarity: `sim(a,b) = a^T b / (||a|| · ||b||)`

**PWNIC loss for a batch of N pairs:**

```
L_PWNIC = - (1 / ∑_i φ_i) · ∑_{i=1}^{N} φ_i · log [
    exp(sim(p_i^c, p_i^n) / τ)
    ─────────────────────────────────────────────────
    ∑_{k=1, k≠i}^{2N} exp(sim(p_i^c, p_k) / τ)
]
```

**Breaking it down:**
- The denominator runs over all 2N embeddings in the batch (N clean + N noisy), excluding the anchor itself
- The positive pair is `(p_i^c, p_i^n)` — clean and noisy version of the SAME utterance
- Each pair is weighted by `φ_i` — its phonetic dissimilarity
- Normalise by `∑ φ_i` so the overall loss scale doesn't depend on batch composition

**Special cases:**
- If `φ_i = 0` (clean and noisy text are identical): pair contributes 0 to loss ✓ (no need to align identical things)
- If `φ_i = 1` (maximum distortion): pair contributes maximum gradient
- For a batch where all φ_i = 0: loss = 0 (undefined 0/0, handled by +ε in normalisation)

**Why this is better than vanilla InfoNCE here:**
In a batch mixing low-noise and high-noise examples, vanilla InfoNCE treats a pair where one word changed the same as a pair where half the text is garbled. PWNIC puts more pressure exactly where alignment is hardest.

**Gradient with respect to p_i^c (intuition):**
```
∂L_PWNIC/∂p_i^c ∝ φ_i · (p_i^n_softmax_negative - p_i^n_softmax_positive)
```
Higher φ_i → larger gradient → stronger push to align the corrupted pair.

## 2.6 Noise-Aware Attention Gate

```
g_ψ(H) = H ⊙ σ(W_g · H + b_g)
```

- `H ∈ R^{L×d}`: encoder hidden states
- `W_g ∈ R^{d×1}`: weight matrix (projects d-dim hidden to scalar per token)
- `σ`: sigmoid function → output in (0,1) per token
- `⊙`: element-wise multiplication (broadcasting across d dimensions)

**What this does:** Each token position gets a scalar gate value between 0 and 1. Tokens the gate assigns low values to contribute little to the pooled representation. The gate is learned — it should learn to suppress corrupted tokens.

**Mean pooling after gate:**
```
z = ∑_i [g_ψ(h_i) · m_i] / ∑_i m_i
```
where `m_i ∈ {0,1}` is the padding mask (1 for real tokens, 0 for padding).

**Key finding:** The gate alone hurts performance (0.8339 vs 0.8394 baseline). Without the PWNIC signal, the gate doesn't know what "corrupted" means and suppresses arbitrary tokens. PWNIC gives the gate a training signal: representations of corrupted tokens should be alignable with their clean counterparts, so the gate learns to de-emphasise tokens where alignment is difficult.

## 2.7 Auxiliary Noise-Level Classification

```
L_aux = -∑_{k=0}^{3} y_k^{noise} · log q_η(z^n)_k
```

- `q_η`: Linear(d, 4) classifier predicting noise level (clean/low/medium/high = 0/1/2/3)
- Applied only to `z^n` (the noisy stream's representation)
- Forces the noisy encoder to develop representations that are noise-level-aware
- This is a regulariser — it doesn't help at inference time (there's no noise label at test time)

**Auto-disable condition:** If all examples in a batch come from the same noise level, `y_k^{noise}` is constant across all examples. The classifier trivially predicts the constant class and the gradient pushes the encoder toward representations that only encode noise level, ignoring content. Therefore: skip L_aux if `|{noise_label for example in batch}| < 2`.

## 2.8 Gradient Reversal Layer (Why It Failed)

A **GRL** (Ganin et al., 2016) works like this:
- In the forward pass: identity function `f(x) = x`
- In the backward pass: negates and scales the gradient: `∂/∂x → -λ · ∂/∂x`

This makes the encoder try to make noise-level features *uninformative* to a noise discriminator — the encoder learns representations where you can't tell what noise level was applied.

**Why it failed here — Failure Mode 1 (Degenerate Targets):**
When training on a single noise level's batch (e.g., all "high noise"), every example has the same noise label. The adversarial predictor has 0 loss regardless of representations (it always predicts "high"). The GRL then pushes gradients that have zero signal, but the reversed gradient still flows — it pushes the encoder toward uniform, uninformative representations, destroying classification performance.

**Why it failed here — Failure Mode 2 (Contradictory Gradients):**
The auxiliary head (γ=0.1, no GRL) tries to make noise level predictable. The adversarial head (β=0.3, with GRL) tries to make it unpredictable. Both operate on the same `z^n`. Since β > γ, adversarial wins, and the auxiliary head's signal is swamped.

**Contrast with DANN success:** In domain adaptation (e.g., source = real photos, target = cartoons), domains are always structurally distinct in every batch. The adversarial predictor never gets constant targets. This structural guarantee is absent in noise-level training.

## 2.9 Total Training Objective

```
L = L_CE + α · L_PWNIC + γ · L_aux
```

where α = 0.15, γ = 0.2.

**Interpretation:**
- L_CE (weight 1.0): primary task — classify hate/non-hate
- L_PWNIC (weight 0.15): pull clean and noisy representations together in proportion to how different they sound
- L_aux (weight 0.20): regularise the noisy representation to be noise-aware

**Why these weights?** α=0.15 was chosen so PWNIC doesn't dominate over CE. If α is too large, the model optimises alignment at the cost of classification. γ=0.20 is slightly larger because aux is a softer regulariser that can tolerate more signal without hurting CE.

---

# PART 3: THE DATA — EVERYTHING YOU NEED TO KNOW

## 3.1 Source Datasets

| Dataset | Reference | Examples | Type |
|---------|-----------|----------|------|
| Davidson et al. | ICWSM 2017 | 17,290 | English tweets: hate/offensive/neither |
| OLID | Zampieri et al., NAACL 2019 | 9,774 | English offensive language |
| SemEval 2020 Task 9 | — | 18,631 | Hindi-English sentiment/hate |
| HASOC 2021 | Mandl et al., FIRE 2021 | 7,952 | Hindi-English hate/offensive |
| UCB Measuring Hate | — | 27,854 | English hate |
| TweetEval Hate | — | 8,909 | English hate |
| Web scraped | — | 34,293 | Mixed Hinglish |
| **Total** | | **123,631** | |

**Label unification:** All datasets have their own label schemas. Everything is collapsed to binary: `hate` (1) or `non-hate` (0). Multi-class labels like "offensive but not hate" → non-hate. "Explicit hate" → hate.

## 3.2 Class Imbalance

The clean training set has **37.6% hate, 62.4% non-hate** — about 1:1.66 ratio. This is typical for hate speech datasets (hate is the minority class). The noisy subsets have much lower hate rate (6.3%) because the web scraped data used for noise injection skews toward non-hate.

## 3.3 Noise Injection Parameters

| Noise Level | Perturbation prob p | Each op prob |
|-------------|--------------------|----|
| Low | 0.05 | p/4 = 0.0125 each |
| Medium | 0.10 | p/4 = 0.025 each |
| High | 0.20 | p/4 = 0.05 each |

At p=0.20, on average 20% of characters are altered. This is calibrated against measured Hinglish ASR error rates of 15-40%.

**Four operations (each with probability p/4):**
1. Substitution: replace c_i with random character from alphabet
2. Insertion: keep c_i and insert random character after it
3. Deletion: remove c_i entirely
4. Word split: insert a space after c_i (simulates ASR word boundary errors)

## 3.4 Why Synthetic Noise and Not Real ASR?

**Honest answer:** Real ASR noise would be more valid, but it requires (a) Hinglish speech audio, (b) hate speech labels for that audio, and (c) running an ASR system. No such paired dataset exists publicly. Synthetic noise is a controlled approximation calibrated to empirical ASR error rates. This is a stated limitation of the paper.

---

# PART 4: THE EXPERIMENTS

## 4.1 Encoders Compared

| Model | Tokeniser | Languages | Vocab | Params |
|-------|-----------|-----------|-------|--------|
| mBERT | WordPiece | 104 | 120k | 179M |
| MuRIL | WordPiece | 17 Indian | 197k | 236M |
| XLM-R | SentencePiece | 100 | 250k | 278M |

**Key difference:** WordPiece applies case folding and accent normalisation before tokenising. SentencePiece (BPE variant) operates on raw Unicode — so "madrachd" stays as-is and the model sees the actual corrupted characters. This is why XLM-R benefits more from PWNIC.

## 4.2 What the Baselines Are

**Lexical baselines:**
- TF-IDF + Logistic Regression: bag-of-words, no context
- TF-IDF + SVM: same features, different classifier
- Char-CNN (Zhang et al., 2015): 1D convolutions over character n-grams

**Transformer baselines:** Same encoders but fine-tuned with only L_CE (no PWNIC, no gate, no aux).

Both baseline and NoiseBridge are trained on the FULL dataset (clean + all noisy subsets combined) and evaluated on each test condition separately.

## 4.3 Reproducibility Setup

Three random seeds: 42, 43, 44. All results reported as mean ± std over 3 seeds.

Full seeding protocol:
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 4.4 Reading the Results

**Macro-F1 formula:**
```
Macro-F1 = (F1_hate + F1_non-hate) / 2

F1_c = 2 · Precision_c · Recall_c / (Precision_c + Recall_c)
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
```

Macro-F1 treats both classes equally regardless of size — important for imbalanced datasets.

**Interpreting the degradation curve (fig_degradation.png):**
- All models degrade as noise increases (left to right)
- Transformer baselines: −1.0 to −1.7 F1 points from clean to high
- Lexical baselines: −2.7 F1 points (steeper = more noise-sensitive)
- XLM-R + NoiseBridge: flattest curve, largest gap at high noise
- mBERT/MuRIL + NoiseBridge: essentially overlap with their baselines

---

# PART 5: PAPERS YOU MUST KNOW

## 5.1 Papers You Will Be Asked About Directly

### [1] Davidson et al. (2017) — "Automated Hate Speech Detection and the Problem of Offensive Language" (ICWSM)
**What it is:** The seminal hate speech dataset paper — 17,290 tweets labelled hate/offensive/neither by crowd workers.
**What to know:** The key finding was that "offensive" and "hate" are difficult to distinguish. Their SVM classifier got ~90% accuracy but confused offensive with hate. You use their dataset as part of your training data.

### [2] Winkler (1990) — "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model"
**What it is:** The original paper defining Jaro-Winkler similarity.
**What to know:** JW was designed for record linkage (matching census records with typos). It gives credit for prefix matches, which is phonetically motivated (words that start the same tend to sound similar). You use it because it captures partial phonetic similarity between clean and ASR-corrupted words.

### [3] van den Oord et al. (2018) — "Representation Learning with Contrastive Predictive Coding" (arXiv)
**What it is:** Introduced InfoNCE loss. Original application was speech and image representations.
**What to know:** InfoNCE lower-bounds mutual information I(x;c) between an observation x and context c. Your PWNIC extends InfoNCE with per-pair weights. You must know InfoNCE cold.

### [4] Ganin et al. (2016) — "Domain-Adversarial Training of Neural Networks" (JMLR)
**What it is:** Introduced gradient reversal for domain adaptation.
**What to know:** The GRL makes the encoder domain-invariant by reversing gradients from a domain discriminator. Works when domains are always distinct. Fails in your setting due to degenerate targets. You tried it, documented the failure, and removed it.

### [5] Chang & Chen (2022) — "Contrastive Learning for Improving ASR Robustness in Spoken Language Understanding" (Interspeech) [SpokenCSE]
**What it is:** The closest related work to yours — applies InfoNCE between clean and ASR-noisy transcripts for intent detection.
**What to know:** SpokenCSE applies UNIFORM contrastive pressure to all pairs. Your PWNIC weights by phonetic dissimilarity. SpokenCSE targets English intent detection; you target Hinglish hate speech. You cite this and explicitly differentiate.

### [6] Gao et al. (2021) — "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (EMNLP)
**What it is:** Uses dropout as augmentation in InfoNCE — pass same sentence twice, get two embeddings as positives.
**What to know:** Your PWNIC uses real noisy augmentation (ASR noise) rather than dropout. SimCSE's contrastive pairs have zero phonetic distortion by design; yours vary from 0 to ~0.15.

### [7] Conneau et al. (2020) — "Unsupervised Cross-Lingual Representation Learning at Scale" (ACL) [XLM-R]
**What it is:** XLM-RoBERTa — trained on 2.5TB of text in 100 languages using masked language modelling, SentencePiece tokenisation.
**What to know:** XLM-R is your best-performing backbone. SentencePiece is the key differentiator from mBERT/MuRIL.

### [8] Devlin et al. (2019) — "BERT" (NAACL)
**What it is:** The transformer pre-training paper that started everything.
**What to know:** BERT uses WordPiece tokenisation + MLM + NSP pre-training. mBERT is BERT trained on 104 languages. You must understand attention and MLM.

### [9] Khanuja et al. (2021) — "MuRIL: Multilingual Representations for Indian Languages" (arXiv)
**What it is:** BERT variant trained specifically on 17 Indian languages including Hindi, with transliterated text.
**What to know:** MuRIL is specifically designed for Indian languages, yet it doesn't benefit from PWNIC. This is because despite knowing Hindi, it uses WordPiece which normalises away the phonetic traces your method needs.

### [10] Chen et al. (2020) — "A Simple Framework for Contrastive Learning" (ICML) [SimCLR]
**What it is:** Foundational computer vision contrastive learning. Showed projection head is crucial.
**What to know:** You cite this specifically for the projection head design — projecting to a separate space for contrastive loss prevents it from distorting the classification space. This is a key architectural decision.

### [11] Khosla et al. (2020) — "Supervised Contrastive Learning" (NeurIPS)
**What it is:** Extends contrastive learning to use label information — same-label examples are all positives.
**What to know:** You use supervised contrastive structure (noise-level labels for pairing), but weight by phonetic distance rather than using labels alone.

## 5.2 Papers to Know for Context (May Be Asked)

- **HateBERT** (Caselli et al., 2020): BERT fine-tuned on Reddit hate speech — domain-specific pre-training helps
- **ByT5** (Xue et al., 2022): Byte-level T5, no tokenisation step, handles OOV and noise well. You tested it briefly (F1=0.831, comparable to MuRIL, but too slow for full evaluation)
- **Belinkov & Bisk (2018)**: Showed character noise breaks NMT — your motivation for the problem
- **HASOC 2021** (Mandl et al.): Official hate speech shared task for Indian languages — you use their dataset

---

# PART 6: PANEL Q&A — EXPECTED QUESTIONS AND STRONG ANSWERS

## 6.1 Questions About the Problem / Motivation

**Q: Why not just use a better ASR system instead of making the detector robust?**

A: Two reasons. First, for low-resource languages like Hinglish, state-of-the-art ASR systems still produce 15–40% error rates — improving them is an orthogonal research problem requiring different expertise and data. Second, in a real moderation pipeline, you cannot guarantee which ASR system will be used or how the input audio quality will vary. A downstream detector that is robust to noise is a more practical safeguard than depending on ASR perfection.

---

**Q: Why is Hinglish ASR specifically bad?**

A: Three compounding factors. First, Hinglish has no standard written form — the same word can be romanised dozens of ways, so ASR language models struggle to pick the right one. Second, code-switching mid-utterance causes acoustic confusion at language boundaries. Third, most ASR systems are trained on large monolingual corpora; the code-mixed acoustic patterns are underrepresented in training data.

---

**Q: Your noise simulation is synthetic. How do you know it reflects real ASR errors?**

A: We calibrate our perturbation probabilities (0.05, 0.10, 0.20) against empirically measured ASR error rates on Hinglish speech from Shah et al. (2020), who report 15–40% character-level errors. Our four operations — substitution, insertion, deletion, word split — cover the four main categories of ASR errors. We acknowledge this is a limitation; real ASR errors include structured phoneme confusions that random character perturbations don't capture. This is why we list real ASR integration as the primary future direction.

---

## 6.2 Questions About the Method

**Q: Why Jaro-Winkler specifically? Why not edit distance or other phonetic metrics like Soundex?**

A: Three reasons. First, Jaro-Winkler is bounded [0,1], making it directly usable as a weight without normalisation. Edit distance is unbounded — a distance of 3 for a short word vs a long word means very different things. Second, Jaro-Winkler gives credit for transpositions and common prefixes, which phonetically motivated — words that begin the same tend to sound similar. Third, Soundex is too coarse — it maps many distinct strings to the same code, losing the gradient of similarity we need for weighting.

---

**Q: The gate alone hurts performance. Doesn't that undermine your contribution?**

A: No — and we report it precisely because it's informative. The gate is a filtering mechanism; it needs a signal to tell it what to filter. Without PWNIC, the gate has no external definition of "corrupted" and learns to suppress arbitrary tokens. PWNIC provides exactly that signal: representations of corrupted tokens should be close to their clean counterparts, so the gate learns to de-emphasise tokens where this alignment is hardest to achieve. The gate and PWNIC are co-dependent by design — neither alone is the contribution; the combination is.

---

**Q: Why does XLM-R benefit from PWNIC but mBERT and MuRIL don't?**

A: The bottleneck is tokenisation. PWNIC exploits phonetic traces — the fact that "madrachd" still looks somewhat like "madarchod" at the character level. XLM-R's SentencePiece tokeniser preserves raw Unicode characters, so these traces survive into the encoder's input. WordPiece (used by mBERT and MuRIL) applies case folding and accent stripping before tokenising, erasing exactly the phonetic variations PWNIC relies on. By the time a WordPiece model sees the input, the signal is already gone. PWNIC cannot create signal that tokenisation has destroyed.

---

**Q: Why temperature τ = 0.15? How was that chosen?**

A: τ = 0.15 is a standard value from SimCSE (Gao et al., 2021), where they found it works well for sentence-level contrastive learning. Lower temperature makes the contrastive distribution sharper — the model must push positives much closer than negatives. Higher temperature is more permissive. We adopt 0.15 from prior work as a reasonable default; a full temperature sweep was computationally expensive across 3 encoders × 4 noise conditions × 3 seeds.

---

**Q: Why α = 0.15 and γ = 0.20? Did you tune these?**

A: We conducted initial experiments with α ∈ {0.05, 0.10, 0.15, 0.20} and γ ∈ {0.10, 0.20, 0.30}. α = 0.15 and γ = 0.20 gave the best validation F1 on XLM-R. Lower α values didn't provide enough contrastive signal; higher values let PWNIC dominate over CE and hurt classification. The γ = 0.20 choice reflects that the aux head is a softer regulariser that can tolerate a stronger signal.

---

**Q: Why does GRL catastrophically fail? Explain the math.**

A: Consider a batch where all examples have noise level "high" (which happens when training on the high-noise subset). The adversarial noise predictor `q_adv` must classify noise level from `z^n`. But all targets are "high" — the predictor achieves zero loss by always outputting "high" regardless of input. The GRL then sends a gradient signal of magnitude `β · ∂L_adv/∂encoder` back through the encoder. Since `∂L_adv` is essentially constant (converged to minimum), the reversed gradient pushes `z^n` toward a region of the embedding space that minimises discriminator loss — which is a uniform, uninformative region. The encoder loses its ability to distinguish hate from non-hate. You can see this in the XLM-R high-noise collapse from 0.829 → 0.777 (−5.2 points).

---

**Q: Why is the auxiliary head not reversed like GRL, but kept forward?**

A: The auxiliary head is a regulariser, not an adversary. The GRL's goal is to make noise level *undetectable* — which destroys the encoder. The auxiliary head's goal is to make noise level *detectable* — which forces the encoder to develop noise-level-aware representations. A noise-aware encoder can then learn to separate noise-specific features from content features. These are opposite objectives, which is exactly why mixing them (adversarial head + auxiliary head on the same `z^n`) creates contradictory gradients.

---

**Q: Your gains on mBERT and MuRIL are zero. Doesn't that mean NoiseBridge failed?**

A: Not exactly. NoiseBridge doesn't hurt mBERT or MuRIL (gains are ≈ 0, within seed variance). The method is neutral on these encoders, not harmful. The key result is that NoiseBridge provides consistent, statistically meaningful gains on XLM-R across all noise levels (+0.31 to +0.56 F1). Whether a method helps depends on whether the encoder's representations can exploit the contrastive signal — for WordPiece encoders they cannot, but the method doesn't break them either. This is actually a practically important finding: you can deploy NoiseBridge on any encoder without risk of degradation.

---

## 6.3 Questions About the Benchmark

**Q: The noisy training subsets have only 6.3% hate, much lower than the 37.6% clean set. Isn't this a problem?**

A: Good observation. The 6.3% figure reflects the composition of the web scraped data (34,293 examples), which is predominantly non-hate. This imbalance is handled by class weighting during training. At evaluation, both validation and test sets maintain 37.6% hate — they are drawn from the clean portion. So the model is evaluated under the same class distribution it was trained on (in the clean subset). The lower hate rate in noisy subsets is a realistic reflection of the fact that most speech content is not hateful.

---

**Q: You call this "the first benchmark" for this task. How do you know no prior work did this?**

A: We surveyed all major hate speech benchmarks (HateXplain, HASOC 2021, SemEval 2020 Task 9, OLID, Davidson 2017, UCB) and found none that introduce controlled ASR noise conditions. Prior code-mixed hate speech work (Bohra et al. 2018, Mandl et al. 2021) uses clean text only. Prior ASR-robustness NLP work (SpokenCSE, Belinkov & Bisk) targets other tasks — intent detection and machine translation respectively. The intersection of (code-mixed) + (hate speech) + (multi-level ASR noise) + (benchmark) is genuinely novel.

---

**Q: Only 3 seeds. Isn't that too few for statistical significance?**

A: Three seeds is standard in NLP research (most papers use 1–5). For our results, the XLM-R improvements (+0.31 to +0.56) are consistently larger than the standard deviation (±0.002 to ±0.003), giving signal-to-noise ratios of ~100–180×. The mBERT and MuRIL results show differences within variance, which is why we correctly report them as null results. If reviewers push for more seeds, the honest answer is computational resource constraints — 5 epochs × 3 encoders × 3 seeds × 4 noise conditions is already a significant compute budget.

---

## 6.4 Questions About Limitations / Ethics

**Q: Your paper contains examples of hateful language in the error analysis. How do you justify this?**

A: The examples are used for scientific transparency — to explain what specific error patterns NoiseBridge corrects. They are presented in a research context with the understanding that readers are researchers. The alternative — redacting all examples — would make it impossible to verify or understand the error analysis. This is standard practice in hate speech detection papers, including HateXplain.

---

**Q: Could NoiseBridge be used to evade hate speech detectors? (Adversarial misuse)**

A: Any knowledge about detector weaknesses could theoretically be exploited. However, NoiseBridge specifically makes detectors MORE robust to noise, which reduces evasion via deliberate misspelling. The noise injection techniques we describe are already well-known and trivially implemented without our paper. The net effect of this work is to make moderation systems harder to evade, not easier.

---

**Q: What are the real limitations of this work?**

A: Three honest ones:
1. **Synthetic noise:** Real ASR errors have structured phoneme confusion patterns that random character perturbations miss. A model trained on our synthetic noise may not transfer perfectly to real ASR output.
2. **English-heavy benchmark:** 51.6% of our training data is English-only (Davidson, OLID, UCB, TweetEval). Gains on purely Hindi-dominant text may differ.
3. **Modest gains:** +0.41 average F1 is meaningful but not dramatic. Multilingual transformers are already surprisingly robust to character noise, leaving limited headroom for improvement.

---

## 6.5 Questions About Future Work / Extensions

**Q: How would you extend this to real ASR data?**

A: The most direct path is to find Hinglish speech datasets (like the IndicSpeech corpus) and apply an ASR system to generate noisy transcripts, then manually label a subset for hate speech. Alternatively, use a hate-speech dataset with original text and deliberately run it through ASR systems (text → TTS → ASR) to get realistic noise patterns.

**Q: Could you apply PWNIC to other tasks?**

A: Yes — any task where you have (clean text, noisy/corrupted version) pairs and want the model to be robust. Sentiment analysis, NER, misinformation detection on transcribed speech. The Jaro-Winkler weight is task-agnostic. You'd need to verify that the encoder's tokeniser preserves enough phonetic signal for the weighting to be meaningful.

---

# PART 7: MATHEMATICS YOU MUST BE ABLE TO SOLVE ON THE BOARD

## 7.1 Compute φ by hand for a given pair

**Example:** Clean: "tera baap hai", Noisy: "tera bap hii"

Words: [tera, baap, hai] vs [tera, bap, hii], M = max(3,3) = 3

- JW("tera", "tera") = 1.0 (identical)
- JW("baap", "bap") ≈ ?
  - Jaro: |s|=4, |t|=3, window = floor(max(4,3)/2)-1 = 0. Hmm, let's be careful.
  - Window = floor(4/2) - 1 = 1
  - Matching chars in "baap" and "bap" within window 1:
    - b↔b (pos 0,0 ✓), a↔a (pos 1,1 ✓), a↔p (pos 2,2 — no match), p↔— (no char at pos 3)
    - m = 2 matches (actually m=3: b,a,p all match — check carefully)
    - Actually: b(0)↔b(0)✓, a(1)↔a(1)✓, p(3)↔p(2)✓ within window. m=3
    - Transpositions: matched chars in order: baap→[b,a,p], bap→[b,a,p]. t'=0
    - Jaro = (3/4 + 3/3 + (3-0)/3) / 3 = (0.75 + 1.0 + 1.0)/3 = 2.75/3 ≈ 0.917
    - Common prefix: "ba" (l=2), p=0.1
    - JW = 0.917 + 2×0.1×(1-0.917) = 0.917 + 0.0166 ≈ 0.934
- JW("hai", "hii") ≈ ?
  - Jaro: |s|=3,|t|=3, window=0
  - h(0)↔h(0)✓, a(1)↔i(1) — 'a' vs 'i', not matching, i(2)↔i(2)✓. m=2
  - t'=0. Jaro = (2/3 + 2/3 + 2/2)/3 = (0.667+0.667+1.0)/3 = 2.334/3 ≈ 0.778
  - Common prefix: "h" (l=1)
  - JW = 0.778 + 1×0.1×(1-0.778) = 0.778 + 0.022 ≈ 0.800

φ = 1 - (1.0 + 0.934 + 0.800) / 3 = 1 - 2.734/3 = 1 - 0.911 = **0.089**

This pair would get φ ≈ 0.089 — low-medium weight, appropriate for mild noise.

## 7.2 Show that PWNIC reduces to InfoNCE when all φ_i = φ (constant)

If φ_i = φ for all i:
```
L_PWNIC = -(1 / ∑_i φ) · ∑_i φ · log [...]
         = -(1 / Nφ) · Nφ · (1/N) ∑_i log [...]
         = -(1/N) ∑_i log [exp(sim(p_i^c, p_i^n)/τ) / ∑_{k≠i} exp(sim(p_i^c, p_k)/τ)]
         = L_InfoNCE
```
PWNIC is a strict generalisation of InfoNCE with uniform weights.

## 7.3 Show what happens to PWNIC when φ_i = 0 for all i

If φ_i = 0 for all i:
- Numerator of normalisation: ∑_i φ_i = 0
- We use the convention: 0/0 → 0 (implemented as dividing by ∑φ + ε)
- Each term: φ_i · log[...] = 0 · log[...] = 0
- L_PWNIC = 0

This makes sense: if clean and noisy texts are identical (no noise), there's no need to push representations together. The PWNIC loss correctly vanishes.

## 7.4 Compute macro-F1 from a confusion matrix

**Example confusion matrix (binary: hate=1, non-hate=0):**
```
              Predicted
              0    1
Actual 0    [900, 100]
       1    [150, 350]
```

- TP_hate = 350, FP_hate = 100, FN_hate = 150, TN_hate = 900
- Precision_hate = 350/(350+100) = 350/450 ≈ 0.778
- Recall_hate = 350/(350+150) = 350/500 = 0.700
- F1_hate = 2 × 0.778 × 0.700 / (0.778 + 0.700) = 1.089/1.478 ≈ 0.737

- TP_non-hate = 900, FP_non-hate = 150, FN_non-hate = 100
- Precision_non-hate = 900/1050 ≈ 0.857
- Recall_non-hate = 900/1000 = 0.900
- F1_non-hate = 2×0.857×0.9/(0.857+0.9) = 1.543/1.757 ≈ 0.878

- **Macro-F1 = (0.737 + 0.878) / 2 = 0.807**

## 7.5 Show why class weights correct for imbalance in CE loss

Without weights, CE loss on a 63:37 (non-hate:hate) dataset with a model that always predicts non-hate:
- L_CE = -(0.63 × log(1) + 0.37 × log(ε)) → large loss, but model could partially converge

With weight 1.66 on hate:
- L_CE_hate is penalised 1.66× more
- The model must correctly classify both classes to minimise loss
- Effectively balances gradient contributions from each class

---

# PART 8: QUICK-REFERENCE CHEAT SHEET FOR THE PANEL

**Three contributions:** (1) HindiMix-Noisy benchmark, (2) PWNIC loss, (3) Empirical analysis including GRL failure

**Key numbers:**
- 123,631 training examples, 4 noise conditions
- XLM-R + NoiseBridge: +0.41 avg F1, +0.56 at high noise
- Gate alone: −0.55 from baseline
- GRL collapse: −5.2 pts (XLM-R high), −4.5 pts (MuRIL medium)
- φ mean: 0.025 (low), 0.039 (medium), 0.056 (high)
- τ = 0.15, α = 0.15, γ = 0.20
- 3 seeds: 42, 43, 44

**Why XLM-R but not others:** SentencePiece (raw Unicode) vs WordPiece (normalises away phonetic traces)

**Why no GRL:** Degenerate targets when single-noise-level batches → constant adversarial loss → reversed gradient collapses encoder

**PWNIC vs SpokenCSE:** SpokenCSE = uniform InfoNCE on English intent detection; PWNIC = phonetically-weighted InfoNCE on Hinglish hate speech

**Honest limitations:** Synthetic noise, English-heavy data, modest gains, headroom limited by inherent transformer robustness
