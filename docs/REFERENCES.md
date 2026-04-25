# References & Citations

All papers used or directly relevant to HindiMix-Noisy.
Format: [Short tag] — used for quick reference in paper writing.

---

## Datasets

[HASOC2021] Modha et al. (2021). "Overview of the HASOC Subtrack at FIRE 2021: Hate Speech and Offensive Content Identification in Indo-European Languages." FIRE 2021.
- Used for: HASOC 2021 CodeMix dataset (2,819 Hindi-English labeled tweets, HOF/NONE)
- URL: https://github.com/AditiBagora/Hasoc2021CodeMix

[SENTIMIX2020] Patwa et al. (2020). "SemEval-2020 Task 9: Sentiment Analysis for Code-Mixed Social Media Text." SemEval 2020.
- Used for: 18,631 Hindi-English Sentimix tweets (used as code-mixed text source)
- URL: https://github.com/singhnivedita/SemEval2020-Task9

[DAVIDSON2017] Davidson et al. (2017). "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM 2017.
- Used for: 24,781 English hate/offensive/neither tweets
- Key finding: hate vs offensive is hard to distinguish — motivates binary label choice

[UCB2020] Kennedy et al. (2020). "Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application." arXiv 2020.
- Used for: 39,565 unique comments with continuous hate_speech_score
- Note: annotator-level data (135K rows) deduplicated by comment_id

[TWEETEVAL] Barbieri et al. (2020). "TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification." EMNLP Findings 2020.
- Used for: 12,713 hate speech tweets (tweeteval/hate subset)

[OLID] Zampieri et al. (2019). "Predicting the Type and Target of Offensive Posts in Social Media." NAACL 2019.
- Used for: 14,052 offensive language identification samples

---

## Related Work: Hate Speech Detection

[WASEEM2016] Waseem & Hovy (2016). "Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter." NAACL SRW 2016.
- First major hate speech dataset on Twitter
- Cite in: Related Work, Dataset section

[FORTUNA2018] Fortuna & Nunes (2018). "A Survey on Automatic Detection of Hate Speech in Text." ACM Computing Surveys.
- Best survey paper on hate speech detection
- Cite in: Introduction, Related Work

[MOZAFARI2020] Mozafari et al. (2020). "A BERT-Based Transfer Learning Approach for Hate Speech Detection in Online Social Media." Complex Networks 2019.
- BERT for hate speech — strong baseline reference
- Cite in: Related Work (transformer baselines)

---

## Related Work: Code-Mixed NLP

[PRATAPA2018] Pratapa et al. (2018). "Language Modeling for Code-Mixing: The Role of Linguistic Theory Based Synthetic Data." ACL 2018.
- Foundational code-mixing NLP paper
- Cite in: Introduction (why code-mixed is hard)

[KHANUJA2020] Khanuja et al. (2020). "GLUECoS: An Evaluation Benchmark for Code-Switched NLP." ACL 2020.
- Benchmark for code-switched tasks including sentiment
- Cite in: Related Work

[MURIL] Khanuja et al. (2021). "MuRIL: Multilingual Representations for Indian Languages." arXiv 2021.
- MuRIL model used as baseline — trained on Indian language data including Hinglish
- Cite in: Experiments (baseline description)

---

## Related Work: Noisy Text / ASR Robustness

[BYT5] Xue et al. (2022). "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models." TACL 2022.
- ByT5: byte-level model, no tokenizer — robust to character-level noise
- Core architectural choice for NoiseBridge
- Key quote: "ByT5 is more robust to noise and spelling errors"

[NAMYSL2020] Namysl et al. (2020). "NAM: Unsupervised Cross-Domain Text Classification via a Novel Noise-Aware Model." arXiv 2020.
- Noise-aware models for text — background for our noise-aware attention gate

[BELINKOV2018] Belinkov & Bisk (2018). "Synthetic and Natural Noise Both Break Neural Machine Translation." ICLR 2018.
- Shows how NLP models break under character-level noise
- Cite in: Introduction (motivation), Related Work

[HEIGOLD2018] Heigold et al. (2018). "Robust Models for Dialect Identification." Interspeech 2018.
- Robustness to dialectal variation in speech — adjacent motivation

---

## Related Work: Contrastive Learning

[SIMCSE] Gao et al. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." EMNLP 2021.
- Standard contrastive learning for sentence embeddings
- What we extend: their InfoNCE → our PWNIC (phonetically weighted)
- Cite in: Model section ("building on SimCSE, we propose...")

[INFONCE] Oord et al. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv 2018.
- Original InfoNCE loss — mathematical foundation of PWNIC
- Cite in: Model section (equation derivation)

[NTXENT] Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
- NT-Xent loss (SimCLR) — variant of InfoNCE we build on
- Cite in: Model section

---

## Related Work: Domain Adversarial / GRL

[DANN] Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR 2016.
- Original GRL paper — our noise disentanglement directly uses their method
- Cite in: Model section ("following Ganin et al., we apply GRL...")
- Key: DANN annealing schedule lambda(p) = 2/(1+exp(-10p)) - 1

[ROUSHAR2020] Roushar et al. (not found) — check if adversarial disentanglement for NLP noise exists
- If no prior work found: strengthens our novelty claim

---

## Multilingual Transformers (Baselines)

[MBERT] Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
- mBERT: multilingual BERT used as baseline

[XLMR] Conneau et al. (2020). "Unsupervised Cross-lingual Representation Learning at Scale." ACL 2020.
- XLM-R used as baseline — stronger than mBERT for cross-lingual tasks

---

## Phonetic Similarity

[JELLYFISH] Jellyfish Python library — JaroWinkler, Soundex, Metaphone implementations
- Used in PWNIC loss computation
- Cite as: software reference

[JARO1989] Jaro (1989). "Advances in Record-Linkage Methodology as Applied to Matching the 1985 Census of Tampa, Florida." Journal of the American Statistical Association.
- Original Jaro similarity — mathematical basis of our phonetic weight phi

[WINKLER1990] Winkler (1990). "String Comparator Metrics and Enhanced Decision Rules in the Fellegi-Sunter Model of Record Linkage."
- Jaro-Winkler extension used in phi computation

---

---
Data Agumentation references:

🔹 A. Data Augmentation in NLP (Core justification)
[EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks]
👉 Shows simple transformations (swap, delete, replace) improve robustness
[Text Data Augmentation for Deep Learning]
👉 Survey paper — VERY useful to justify your whole approach

🔹 B. Backtranslation / Paraphrasing
[Improving Neural Machine Translation Models with Monolingual Data]
👉 Introduced backtranslation (widely accepted)
[Unsupervised Data Augmentation for Consistency Training]
👉 Shows augmentation improves robustness

🔹 C. Noise Injection / Robustness
[On the Robustness of NLP Models to Input Perturbations]
👉 Character noise + perturbations (exactly what you do)
[TextFlint: Unified Multilingual Robustness Evaluation Toolkit]
👉 Supports noise-based robustness evaluation

🔹 D. Code-Mixing / Hinglish
[Sentiment Analysis of Code-Mixed Languages leveraging Resource Rich Languages]
[A Benchmark Dataset for Code-Mixed Hinglish Text Classification]
👉 These justify:
Hinglish transformations
code-mixed processing

🔹 E. Class Imbalance Handling
[SMOTE: Synthetic Minority Over-sampling Technique]
👉 Even though it's tabular:
concept of oversampling minority class = exactly what you do
🧠 2. How YOUR method maps to literature
Your Method   	           Paper Support
Character noise	           Belinkov & Bisk
Hinglish transformation	   Code-mixing papers
Oversampling hate	       SMOTE
Synthetic augmentation	   EDA, Feng survey
Robustness training	       UDA, TextFlint


## To Find / Verify

- [ ] Any paper doing contrastive learning for noisy Hinglish — if none found, strong novelty claim
- [ ] Any paper applying GRL to noise robustness (not domain) — if none found, strong novelty claim
- [ ] HASOC 2019/2020 papers (earlier years, cite as dataset context)
- [ ] LinCE benchmark paper (Aguilar et al. 2020) — code-mixed evaluation benchmark
