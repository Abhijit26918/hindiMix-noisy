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
    """HASOC 2021 CodeMix: HOF (hate/offensive) → 1, NOT → 0.
    Handles multiple possible column name formats across HASOC years."""
    df = pd.read_csv(path)
    explore(df, "HASOC")

    # Text column — try common names
    text_col = next((c for c in df.columns if any(k in c.lower() for k in
                     ["text", "tweet", "content", "post", "sentence"])), df.columns[0])

    # Label column — task_1 is the HOF/NOT binary task
    label_col = next((c for c in df.columns if any(k in c.lower() for k in
                      ["task_1", "label", "task1", "hof", "class"])), None)
    if label_col is None:
        label_col = df.columns[1]

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].map(
        lambda x: 1 if str(x).upper().strip() in ["HOF", "1", "HATE", "OFF", "OFFENSIVE", "TRUE"] else 0
    )
    out["source"] = "hasoc2021"
    out["lang_pair"] = "hi-en"
    return out[out["text"].str.len() > 5].dropna()


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
    """SemEval 2020 Task 9 Sentimix — Hindi-English code-mixed sentiment.
    Format (from singhnivedita/SemEval2020-Task9): uid | text | label(0/1/2)
    Sentiment labels (not hate): used as code-mixed text source → label=0 (non-hate).
    """
    df = pd.read_csv(path)
    explore(df, "SemEval Sentimix")
    out = pd.DataFrame()
    # Handle both processed TSV format (uid, text, label) and other formats
    if "text" in df.columns:
        out["text"] = df["text"].astype(str)
    elif "tweet" in df.columns:
        out["text"] = df["tweet"].astype(str)
    else:
        out["text"] = df.iloc[:, 1].astype(str)  # second column is text in TSV format
    out["label"] = 0  # sentiment data used as non-hate code-mixed source
    out["source"] = "semeval2020"
    out["lang_pair"] = "hi-en"
    return out[out["text"].str.len() > 5].dropna()


def adapt_hindi_offensive(path):
    """Hindi offensive dataset: map hate/offensive → 1, non-hate → 0."""
    df = pd.read_csv(path)
    explore(df, "Hindi Offensive")
    out = pd.DataFrame()
    text_col = next((c for c in df.columns if any(k in c.lower() for k in ["text", "tweet", "sentence", "content"])), df.columns[0])
    label_col = next((c for c in df.columns if any(k in c.lower() for k in ["label", "class", "category", "hate"])), df.columns[1])
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].map(lambda x: 1 if str(x).upper() in ["1", "HOF", "HATE", "OFFENSIVE", "OFF", "TRUE"] else 0)
    out["source"] = "hindi_offensive"
    out["lang_pair"] = "hi-en"
    return out.dropna()


def adapt_ucb_hate(path):
    """UC Berkeley Measuring Hate Speech.
    Dataset is annotator-level (multiple rows per comment) — aggregate first.
    hate_speech_score >= 0.5 = hate (threshold from original paper)."""
    df = pd.read_csv(path)
    explore(df, "UCB Measuring Hate Speech")

    # Aggregate multiple annotators per comment → mean hate_speech_score
    if "comment_id" in df.columns and "hate_speech_score" in df.columns:
        agg = df.groupby("comment_id").agg(
            text=("text", "first"),
            hate_speech_score=("hate_speech_score", "mean")
        ).reset_index()
        print(f"  Deduplicated: {len(df):,} annotator rows → {len(agg):,} unique comments")
        df = agg
        label_series = (df["hate_speech_score"] >= 0.5).astype(int)
        text_series = df["text"].astype(str)
    else:
        text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
        text_series = df[text_col].astype(str)
        label_series = (df["hate_speech_score"] >= 0.5).astype(int) if "hate_speech_score" in df.columns else 0

    out = pd.DataFrame()
    out["text"] = text_series
    out["label"] = label_series
    out["source"] = "ucb_measuring_hate"
    out["lang_pair"] = "en"
    return out.dropna()


def adapt_olid(path):
    """OLID: subtask_a OFF=offensive → 1, NOT → 0."""
    df = pd.read_csv(path)
    explore(df, "OLID")
    out = pd.DataFrame()
    text_col = next((c for c in df.columns if "tweet" in c.lower() or "text" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "subtask_a" in c.lower() or "label" in c.lower()), df.columns[1])
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].map(lambda x: 1 if str(x).upper() in ["OFF","1","OFFENSIVE","HATE"] else 0)
    out["source"] = "olid"
    out["lang_pair"] = "en"
    return out.dropna()


def adapt_hateval(path):
    """HatEval 2019: HS=1 hate, HS=0 not hate."""
    df = pd.read_csv(path)
    explore(df, "HatEval 2019")
    out = pd.DataFrame()
    text_col = next((c for c in df.columns if "text" in c.lower() or "tweet" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "hs" in c.lower() or "label" in c.lower()), df.columns[1])
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].map(lambda x: 1 if str(x) in ["1","True","hate"] else 0)
    out["source"] = "hateval2019"
    out["lang_pair"] = "en"
    return out.dropna()


def adapt_hatexplain(path):
    """HateXplain: annotated_span format — majority label across annotators."""
    df = pd.read_csv(path)
    explore(df, "HateXplain")
    out = pd.DataFrame()
    # text is stored as a list of tokens in 'post_tokens' column
    if "post_tokens" in df.columns:
        out["text"] = df["post_tokens"].astype(str).str.strip("[]").str.replace("'", "").str.replace(",", "")
    else:
        text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
        out["text"] = df[text_col].astype(str)
    # label column: hatespeech/offensive/normal
    label_col = next((c for c in df.columns if "label" in c.lower()), None)
    if label_col:
        out["label"] = df[label_col].map(lambda x: 1 if str(x).lower() in ["hatespeech", "offensive", "1", "hate"] else 0)
    else:
        out["label"] = 0
    out["source"] = "hatexplain"
    out["lang_pair"] = "en"
    return out.dropna()


def adapt_tweeteval_hate(path):
    """TweetEval-Hate: label 1=hate, 0=not hate."""
    df = pd.read_csv(path)
    explore(df, "TweetEval-Hate")
    out = pd.DataFrame()
    text_col = next((c for c in df.columns if "text" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "label" in c.lower()), df.columns[1])
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].map(lambda x: 1 if str(x) in ["1", "hate"] else 0)
    out["source"] = "tweeteval_hate"
    out["lang_pair"] = "en"
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
            elif "hindi_offensive" in fname:
                dfs.append(adapt_hindi_offensive(path))
            elif "hatexplain" in fname:
                dfs.append(adapt_hatexplain(path))
            elif "tweeteval" in fname:
                dfs.append(adapt_tweeteval_hate(path))
            elif "ucb_measuring" in fname:
                dfs.append(adapt_ucb_hate(path))
            elif "olid" in fname:
                dfs.append(adapt_olid(path))
            elif "hateval" in fname:
                dfs.append(adapt_hateval(path))
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
