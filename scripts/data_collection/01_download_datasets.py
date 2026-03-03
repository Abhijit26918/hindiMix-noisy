"""
Script 1: Download & collect source hate speech datasets.

Sources used:
  1. HASOC 2019/2020/2021 (Hindi-English code-mixed)  — HuggingFace / manual
  2. SemEval 2020 Task 9 (Sentimix — code-mixed)       — HuggingFace
  3. Hate Speech & Offensive Language (Davidson 2017)  — HuggingFace
  4. CONSTRAINT 2021 (COVID fake news, code-mixed)     — HuggingFace

Run: python scripts/data_collection/01_download_datasets.py
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def log(msg):
    print(f"[INFO] {msg}")


# ─────────────────────────────────────────────────────────────────
# 1. SemEval 2020 Task 9 — Sentiment (code-mixed, Hindi-English)
#    We'll use this for the code-mixed text distribution,
#    and apply hate-speech labels from HASOC.
# ─────────────────────────────────────────────────────────────────
def download_semeval_sentimix():
    log("Downloading SemEval 2020 Sentimix (Hindi-English code-mixed)...")
    try:
        ds = load_dataset("dair-ai/sentimix-hindi-english", trust_remote_code=True)
        df = pd.DataFrame(ds["train"])
        df.to_csv(f"{RAW_DIR}/semeval2020_sentimix.csv", index=False)
        log(f"  Saved {len(df)} samples to {RAW_DIR}/semeval2020_sentimix.csv")
    except Exception as e:
        log(f"  Could not auto-download SemEval: {e}")
        log("  Manual download: https://ritual.uh.edu/lince/datasets")


# ─────────────────────────────────────────────────────────────────
# 2. HASOC 2021 — Hate Speech, code-mixed Hindi-English
# ─────────────────────────────────────────────────────────────────
def download_hasoc():
    log("Downloading HASOC 2021 dataset...")
    try:
        ds = load_dataset("hasoc/hasoc2021", trust_remote_code=True)
        for split_name, split_data in ds.items():
            df = pd.DataFrame(split_data)
            df.to_csv(f"{RAW_DIR}/hasoc2021_{split_name}.csv", index=False)
            log(f"  Saved {len(df)} samples ({split_name}) to {RAW_DIR}/hasoc2021_{split_name}.csv")
    except Exception as e:
        log(f"  HASOC auto-download failed: {e}")
        log("  Manual: https://hasocfire.github.io/hasoc/2021/dataset.html")


# ─────────────────────────────────────────────────────────────────
# 3. Hate Speech Offensive Language (Davidson 2017) — English
#    Used for English-side augmentation
# ─────────────────────────────────────────────────────────────────
def download_davidson():
    log("Downloading Davidson hate speech dataset...")
    try:
        ds = load_dataset("tdavidson/hate_speech_offensive", trust_remote_code=True)
        df = pd.DataFrame(ds["train"])
        df.to_csv(f"{RAW_DIR}/davidson_hate_speech.csv", index=False)
        log(f"  Saved {len(df)} samples to {RAW_DIR}/davidson_hate_speech.csv")
    except Exception as e:
        log(f"  Davidson download failed: {e}")


# ─────────────────────────────────────────────────────────────────
# 4. Hindi Offensive (HuggingFace)
# ─────────────────────────────────────────────────────────────────
def download_hindi_offensive():
    log("Downloading Hindi Offensive dataset...")
    try:
        ds = load_dataset("hate_speech18", trust_remote_code=True)
        df = pd.DataFrame(ds["train"])
        df.to_csv(f"{RAW_DIR}/hindi_offensive.csv", index=False)
        log(f"  Saved {len(df)} samples to {RAW_DIR}/hindi_offensive.csv")
    except Exception as e:
        log(f"  Hindi offensive download failed: {e}")


# ─────────────────────────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────────────────────────
def print_summary():
    log("\n=== Download Summary ===")
    for fname in os.listdir(RAW_DIR):
        if fname.endswith(".csv"):
            path = os.path.join(RAW_DIR, fname)
            df = pd.read_csv(path)
            log(f"  {fname}: {len(df)} rows, columns: {list(df.columns)}")


if __name__ == "__main__":
    download_semeval_sentimix()
    download_hasoc()
    download_davidson()
    download_hindi_offensive()
    print_summary()
    log("\nNext step: Run 02_explore_and_merge.py")
