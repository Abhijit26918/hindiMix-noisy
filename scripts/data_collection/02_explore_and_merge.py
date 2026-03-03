"""
Script 2: Explore downloaded datasets, unify schema, and merge.

After running this you get: data/processed/merged_clean.csv
Columns: [text, label, source, lang_pair]
  label: 0 = non-hate, 1 = hate/offensive
  lang_pair: hi-en (Hindi-English), en (English), hi (Hindi)

Run: python scripts/data_collection/02_explore_and_merge.py
"""

import os
import pandas as pd
import numpy as np
from collections import Counter

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)


def explore(df, name):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample:\n{df.head(2).to_string()}")
    print(f"  Nulls: {df.isnull().sum().to_dict()}")


# ─────────────────────────────────────────────────────────────────
# Adapters: normalize each dataset into unified schema
# ─────────────────────────────────────────────────────────────────

def adapt_hasoc(path):
    """HASOC labels: HOF (hate/offensive) → 1, NOT → 0"""
    df = pd.read_csv(path)
    explore(df, "HASOC")

    # Find text and label columns (may vary by year)
    text_col = next((c for c in df.columns if 'text' in c.lower() or 'tweet' in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if 'label' in c.lower() or 'task' in c.lower()), df.columns[1])

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].map(lambda x: 1 if str(x).upper() in ["HOF", "1", "HATE", "OFF"] else 0)
    out["source"] = "hasoc2021"
    out["lang_pair"] = "hi-en"
    return out.dropna()


def adapt_davidson(path):
    """Davidson: class 0=hate, 1=offensive, 2=neither → we map 0&1 → hate"""
    df = pd.read_csv(path)
    explore(df, "Davidson")
    out = pd.DataFrame()
    out["text"] = df["tweet"].astype(str)
    out["label"] = df["class"].map(lambda x: 1 if x in [0, 1] else 0)
    out["source"] = "davidson"
    out["lang_pair"] = "en"
    return out.dropna()


def adapt_semeval(path):
    """SemEval Sentimix: sentiment labels → not hate labels.
    We use this only for code-mixed text samples (label as non-hate = 0).
    Real hate samples come from HASOC."""
    df = pd.read_csv(path)
    explore(df, "SemEval Sentimix")
    out = pd.DataFrame()
    # Column may be 'tweet' or 'text'
    text_col = "tweet" if "tweet" in df.columns else df.columns[0]
    out["text"] = df[text_col].astype(str)
    out["label"] = 0  # used as non-hate code-mixed source
    out["source"] = "semeval2020"
    out["lang_pair"] = "hi-en"
    return out.dropna()


# ─────────────────────────────────────────────────────────────────
# Merge and balance
# ─────────────────────────────────────────────────────────────────

def merge_all():
    dfs = []

    for fname in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, fname)
        if not fname.endswith(".csv"):
            continue
        try:
            if "hasoc" in fname:
                dfs.append(adapt_hasoc(path))
            elif "davidson" in fname:
                dfs.append(adapt_davidson(path))
            elif "semeval" in fname or "sentimix" in fname:
                dfs.append(adapt_semeval(path))
        except Exception as e:
            print(f"  [WARN] Could not adapt {fname}: {e}")

    if not dfs:
        print("[ERROR] No datasets found. Run 01_download_datasets.py first.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset="text")
    merged = merged[merged["text"].str.len() > 5]  # filter very short texts

    print(f"\n{'='*50}")
    print(f"MERGED DATASET")
    print(f"  Total: {len(merged)}")
    print(f"  Label distribution: {Counter(merged['label'])}")
    print(f"  Sources: {Counter(merged['source'])}")
    print(f"  Lang pairs: {Counter(merged['lang_pair'])}")

    # Save full merged
    merged.to_csv(f"{PROC_DIR}/merged_clean.csv", index=False)
    print(f"\n  Saved to {PROC_DIR}/merged_clean.csv")

    # Save code-mixed subset separately (for our main task)
    codemixed = merged[merged["lang_pair"] == "hi-en"]
    codemixed.to_csv(f"{PROC_DIR}/codemixed_clean.csv", index=False)
    print(f"  Code-mixed subset: {len(codemixed)} samples → {PROC_DIR}/codemixed_clean.csv")

    return merged


if __name__ == "__main__":
    df = merge_all()
    if df is not None:
        print("\nNext step: Run scripts/noise_generation/03_add_noise.py")
