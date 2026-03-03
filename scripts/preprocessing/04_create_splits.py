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
    # Load clean data
    clean_path = os.path.join(PROC_DIR, "codemixed_clean.csv")
    if not os.path.exists(clean_path):
        clean_path = os.path.join(PROC_DIR, "merged_clean.csv")
    if not os.path.exists(clean_path):
        print("[ERROR] No processed data. Run scripts 01-03 first.")
        return

    df = pd.read_csv(clean_path)
    print(f"[INFO] Total clean samples: {len(df)}")
    print(f"  Label distribution: {Counter(df['label'])}")

    # Stratified split
    train_val, test = train_test_split(df, test_size=TEST_RATIO, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(train_val, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), stratify=train_val["label"], random_state=SEED)

    print(f"\n[INFO] Split sizes:")
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Add noise_level column
    train["noise_level"] = "clean"
    val["noise_level"] = "clean"
    test["noise_level"] = "clean"

    # For train: also include ALL noisy versions (data augmentation)
    noisy_parts = [train]
    for level in ["low", "medium", "high"]:
        noisy_path = f"{NOISY_DIR}/{level}/noisy_{level}.csv"
        if os.path.exists(noisy_path):
            noisy_df = pd.read_csv(noisy_path)
            # Only include rows that are in train (by index)
            train_indices = train.index
            noisy_train = noisy_df.loc[noisy_df.index.isin(train_indices)].copy()
            noisy_train["noise_level"] = level
            noisy_parts.append(noisy_train)

    train_full = pd.concat(noisy_parts, ignore_index=True).sample(frac=1, random_state=SEED)

    # Save splits
    train_full.to_csv(f"{FINAL_DIR}/train.csv", index=False)
    val.to_csv(f"{FINAL_DIR}/val.csv", index=False)
    test.to_csv(f"{FINAL_DIR}/test_clean.csv", index=False)

    # Save noisy test sets (same test rows, but noisy)
    for level in ["low", "medium", "high"]:
        noisy_path = f"{NOISY_DIR}/{level}/noisy_{level}.csv"
        if os.path.exists(noisy_path):
            noisy_df = pd.read_csv(noisy_path)
            noisy_test = noisy_df.loc[noisy_df.index.isin(test.index)].copy()
            noisy_test.to_csv(f"{FINAL_DIR}/test_noisy_{level}.csv", index=False)
            print(f"  test_noisy_{level}.csv: {len(noisy_test)} samples")

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
