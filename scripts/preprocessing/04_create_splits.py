"""
Script 4: Create final train/val/test splits for all noise levels.

Output structure in data/final/:
  train.csv       — 70% clean + noisy
  val.csv         — 15% clean + noisy
  test.csv        — 15% clean (held-out clean test)
  test_noisy_low.csv
  test_noisy_medium.csv
  test_noisy_high.csv

Run: python scripts/preprocessing/04_create_splits.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

PROC_DIR = "data/processed"
NOISY_DIR = "data/noisy"
FINAL_DIR = "data/final"
os.makedirs(FINAL_DIR, exist_ok=True)

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_splits():
    # Load merged_clean.csv for train/val/test splits.
    # Reason: codemixed_clean.csv has no hate labels (SemEval is sentiment only).
    # merged_clean.csv has balanced labels from Davidson + TweetEval + UCB + OLID.
    # Noisy test sets use codemixed (Hinglish) data for cross-lingual robustness eval.
    clean_path = os.path.join(PROC_DIR, "merged_clean.csv")
    if not os.path.exists(clean_path):
        print("[ERROR] merged_clean.csv not found. Run scripts 01-02 first.")
        return

    df = pd.read_csv(clean_path)

    if len(df) == 0:
        print("[ERROR] merged_clean.csv is empty.")
        return
    print(f"[INFO] Total clean samples: {len(df)}")
    print(f"  Label distribution: {Counter(df['label'])}")

    # Stratified split
    train_val, test = train_test_split(df, test_size=TEST_RATIO, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(train_val, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), stratify=train_val["label"], random_state=SEED)

    print(f"\n[INFO] Split sizes:")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Reset index for clean joins
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    train["noise_level"] = "clean"
    val["noise_level"]   = "clean"
    test["noise_level"]  = "clean"

    # For train: augment with noisy versions of train texts (from merged data)
    train_texts = set(train["text"])
    noisy_parts = [train]
    for level in ["low", "medium", "high"]:
        noisy_path = f"{NOISY_DIR}/{level}/noisy_{level}.csv"
        if os.path.exists(noisy_path):
            noisy_df = pd.read_csv(noisy_path)
            if "text_original" in noisy_df.columns:
                noisy_train = noisy_df[noisy_df["text_original"].isin(train_texts)].copy()
            else:
                noisy_train = noisy_df.copy()
            noisy_train["noise_level"] = level
            noisy_parts.append(noisy_train)
            print(f"  Added {len(noisy_train):,} {level}-noise augmentation samples")

    train_full = pd.concat(noisy_parts, ignore_index=True).sample(frac=1, random_state=SEED)

    # Save main splits
    train_full.to_csv(f"{FINAL_DIR}/train.csv", index=False)
    val.to_csv(f"{FINAL_DIR}/val.csv", index=False)
    test.to_csv(f"{FINAL_DIR}/test_clean.csv", index=False)

    # Noisy test sets — apply noise directly to test_clean (balanced labels)
    # This gives proper evaluation: same texts, same labels, just with ASR noise added
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location("noise_module", "scripts/noise_generation/03_add_noise.py")
    noise_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(noise_module)
    add_noise = noise_module.add_noise

    print(f"\n[INFO] Generating noisy test sets from test_clean ({len(test):,} rows)...")
    for level in ["low", "medium", "high"]:
        noisy_texts = [add_noise(str(t), level) for t in test["text"]]
        noisy_test = test.copy()
        noisy_test["text_original"] = test["text"]
        noisy_test["text"] = noisy_texts
        noisy_test["noise_level"] = level
        noisy_test.to_csv(f"{FINAL_DIR}/test_noisy_{level}.csv", index=False)
        labels = dict(Counter(noisy_test["label"].astype(int)))
        print(f"  test_noisy_{level}.csv: {len(noisy_test):,} samples | labels: {labels}")

    print(f"\n[INFO] All splits saved to {FINAL_DIR}/")
    print(f"  train.csv: {len(train_full)} rows (clean + augmented noisy)")
    print(f"  val.csv:   {len(val)} rows")
    print(f"  test_clean.csv: {len(test)} rows")

    # Dataset stats report
    report = {
        "total_clean": len(df),
        "train_size": len(train_full),
        "val_size": len(val),
        "test_size": len(test),
        "hate_ratio_train": (train_full["label"] == 1).mean(),
        "hate_ratio_test": (test["label"] == 1).mean(),
    }
    pd.Series(report).to_csv(f"{FINAL_DIR}/dataset_stats.csv")
    print(f"\n[INFO] Stats saved to {FINAL_DIR}/dataset_stats.csv")
    print("\n[INFO] Next: Run Phase 2 baselines → models/baselines/")


if __name__ == "__main__":
    create_splits()
