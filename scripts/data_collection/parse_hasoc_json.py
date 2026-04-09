"""
Parse HASOC 2021 CodeMix dataset from nested JSON structure.

Structure:
  {split}/{topic}/{tweet_id}/data.json    — tweet thread (root + comments + replies)
  {split}/{topic}/{tweet_id}/labels.json  — {tweet_id: "HOF"/"NONE"} for every tweet

Output: data/raw/hasoc2021_train.csv, data/raw/hasoc2021_test.csv

Usage:
    python scripts/data_collection/parse_hasoc_json.py
"""

import os
import json
import pandas as pd
from pathlib import Path

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

HASOC_ROOT = None  # auto-detected below


def find_hasoc_root():
    """Auto-detect HASOC dataset root directory."""
    candidates = [
        "Hasoc2021CodeMix-main",
        "data/hasoc",
    ]
    for c in candidates:
        for root, dirs, files in os.walk(c):
            if "train" in dirs and any(
                os.path.isdir(os.path.join(root, "train", d))
                for d in os.listdir(os.path.join(root, "train"))
                if os.path.isdir(os.path.join(root, "train", d))
            ):
                return root
    # deeper search
    for root, dirs, files in os.walk("."):
        if "train" in dirs:
            train_path = os.path.join(root, "train")
            subdirs = [d for d in os.listdir(train_path)
                       if os.path.isdir(os.path.join(train_path, d))]
            if subdirs:
                # check if it contains tweet_id folders with data.json
                first_topic = os.path.join(train_path, subdirs[0])
                thread_dirs = os.listdir(first_topic)
                if thread_dirs and os.path.exists(
                    os.path.join(first_topic, thread_dirs[0], "data.json")
                ):
                    return root
    return None


def extract_tweets_from_thread(data, labels):
    """Recursively extract all tweets from a thread and join with labels."""
    rows = []

    def recurse(node):
        tid = str(node.get("tweet_id", ""))
        text = node.get("tweet", "")
        label_str = labels.get(tid, "NONE")
        label = 1 if label_str == "HOF" else 0

        if text and len(text.strip()) > 3:
            rows.append({
                "tweet_id": tid,
                "text": text.strip(),
                "label": label,
                "label_str": label_str,
            })

        for comment in node.get("comments", []):
            recurse(comment)
            for reply in comment.get("replies", []):
                recurse(reply)

    recurse(data)
    return rows


def parse_split(split_dir, split_name):
    """Parse all threads in a split directory."""
    all_rows = []
    split_path = Path(split_dir)

    topics = [d for d in split_path.iterdir() if d.is_dir()]
    print(f"\n  Topics found: {[t.name for t in topics]}")

    for topic_dir in topics:
        topic_name = topic_dir.name
        thread_dirs = [d for d in topic_dir.iterdir() if d.is_dir()]

        for thread_dir in thread_dirs:
            data_file   = thread_dir / "data.json"
            labels_file = thread_dir / "labels.json"

            if not data_file.exists() or not labels_file.exists():
                continue

            try:
                with open(data_file,   encoding="utf-8") as f:
                    data = json.load(f)
                with open(labels_file, encoding="utf-8") as f:
                    labels = json.load(f)

                rows = extract_tweets_from_thread(data, labels)
                for r in rows:
                    r["topic"] = topic_name
                    r["split"] = split_name
                all_rows.extend(rows)

            except Exception as e:
                print(f"    [WARN] Failed {thread_dir}: {e}")

    return all_rows


def main():
    global HASOC_ROOT

    HASOC_ROOT = find_hasoc_root()
    if not HASOC_ROOT:
        print("[ERROR] Could not find HASOC dataset directory.")
        print("  Expected structure: Hasoc2021CodeMix-main/.../data/train/{topic}/{tweet_id}/data.json")
        return

    print(f"[INFO] Found HASOC root: {HASOC_ROOT}")

    # Find train and test directories
    train_dir = None
    test_dir  = None
    for root, dirs, files in os.walk(HASOC_ROOT):
        if os.path.basename(root) == "train":
            train_dir = root
        if os.path.basename(root) == "test":
            test_dir = root

    total_saved = 0

    # Parse train
    if train_dir:
        print(f"\n[INFO] Parsing train: {train_dir}")
        rows = parse_split(train_dir, "train")
        df = pd.DataFrame(rows).drop_duplicates(subset="tweet_id")
        out = os.path.join(RAW_DIR, "hasoc2021_train.csv")
        df.to_csv(out, index=False)
        hof  = (df["label"] == 1).sum()
        none = (df["label"] == 0).sum()
        print(f"\n  Train: {len(df):,} tweets | HOF={hof:,} | NONE={none:,}")
        print(f"  Saved -> {out}")
        total_saved += len(df)

        # Topic breakdown
        print(f"  Topics: {dict(df['topic'].value_counts())}")
    else:
        print("[WARN] No train directory found.")

    # Parse test
    if test_dir:
        print(f"\n[INFO] Parsing test: {test_dir}")
        rows = parse_split(test_dir, "test")
        df = pd.DataFrame(rows).drop_duplicates(subset="tweet_id")
        out = os.path.join(RAW_DIR, "hasoc2021_test.csv")
        df.to_csv(out, index=False)
        hof  = (df["label"] == 1).sum()
        none = (df["label"] == 0).sum()
        print(f"\n  Test: {len(df):,} tweets | HOF={hof:,} | NONE={none:,}")
        print(f"  Saved -> {out}")
        total_saved += len(df)
    else:
        print("[WARN] No test directory found.")

    print(f"\n[INFO] Done. Total saved: {total_saved:,} tweets")
    print("[INFO] Next: Run scripts/data_collection/02_explore_and_merge.py")


if __name__ == "__main__":
    main()
