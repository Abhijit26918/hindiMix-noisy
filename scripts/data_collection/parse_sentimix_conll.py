"""
Parse SemEval 2020 Task 9 SentiMix CONLL files → CSV.

CONLL Format:
    meta    uid    sentiment
    token   lang_id
    token   lang_id
    ...
    (blank line between tweets)

lang_id: HIN, ENG, O (neither)
sentiment: positive, negative, neutral

Usage:
    python scripts/data_collection/parse_sentimix_conll.py
    python scripts/data_collection/parse_sentimix_conll.py --train path/to/train.conll --test path/to/test.conll --labels path/to/test_labels.txt
"""

import os
import argparse
import pandas as pd

RAW_DIR = "data/raw"


def parse_conll(filepath, label_file=None):
    """Parse a SentiMix CONLL file into a DataFrame."""
    records = []
    current_uid = None
    current_sentiment = None
    current_tokens = []
    current_langs = []

    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\n")

        if line.startswith("meta"):
            # Save previous tweet if exists
            if current_uid is not None and current_tokens:
                records.append({
                    "uid": current_uid,
                    "text": " ".join(current_tokens),
                    "sentiment": current_sentiment,
                    "lang_ids": " ".join(current_langs),
                    "hindi_ratio": current_langs.count("HIN") / max(len(current_langs), 1),
                })
            # Parse new tweet header: meta  uid  sentiment
            parts = line.split()
            current_uid = parts[1] if len(parts) > 1 else None
            current_sentiment = parts[2].lower() if len(parts) > 2 else None
            current_tokens = []
            current_langs = []

        elif line.strip() == "":
            continue  # blank line between tweets

        else:
            # Token line: token  lang_id
            parts = line.split()
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_langs.append(parts[1])
            elif len(parts) == 1:
                current_tokens.append(parts[0])
                current_langs.append("O")

    # Save last tweet
    if current_uid is not None and current_tokens:
        records.append({
            "uid": current_uid,
            "text": " ".join(current_tokens),
            "sentiment": current_sentiment,
            "lang_ids": " ".join(current_langs),
            "hindi_ratio": current_langs.count("HIN") / max(len(current_langs), 1),
        })

    df = pd.DataFrame(records)

    # If separate label file provided (for test set)
    if label_file and os.path.exists(label_file):
        labels = pd.read_csv(label_file, sep="\t", header=None, names=["uid", "sentiment"])
        df = df.drop(columns=["sentiment"], errors="ignore")
        df = df.merge(labels, on="uid", how="left")

    return df


def conll_to_pipeline_csv(df, output_path, source_name="semeval2020"):
    """Convert parsed CONLL df to the unified pipeline schema."""
    out = pd.DataFrame()
    out["text"] = df["text"].astype(str)
    out["label"] = 0          # SentiMix is sentiment, not hate — used as non-hate source
    out["source"] = source_name
    out["lang_pair"] = "hi-en"

    out = out[out["text"].str.len() > 5]
    out = out.dropna()
    out.to_csv(output_path, index=False)

    print(f"[INFO] Saved {len(out):,} samples → {output_path}")
    print(f"  Sample tweets:")
    for t in df["text"].head(3):
        print(f"    {t}")


def main(args):
    os.makedirs(RAW_DIR, exist_ok=True)

    all_dfs = []

    # Parse train file
    if args.train and os.path.exists(args.train):
        print(f"[INFO] Parsing train: {args.train}")
        df_train = parse_conll(args.train)
        print(f"  Found {len(df_train):,} tweets")
        print(f"  Sentiment dist: {df_train['sentiment'].value_counts().to_dict()}")
        print(f"  Avg Hindi ratio: {df_train['hindi_ratio'].mean():.2%}")
        all_dfs.append(df_train)
    else:
        # Auto-detect in data/raw/
        for fname in os.listdir(RAW_DIR):
            if "train" in fname and fname.endswith(".conll"):
                path = os.path.join(RAW_DIR, fname)
                print(f"[INFO] Auto-detected: {path}")
                all_dfs.append(parse_conll(path))

    # Parse test file + labels
    if args.test and os.path.exists(args.test):
        print(f"[INFO] Parsing test: {args.test}")
        df_test = parse_conll(args.test, label_file=args.labels)
        print(f"  Found {len(df_test):,} test tweets")
        all_dfs.append(df_test)

    if not all_dfs:
        print("[ERROR] No CONLL files found.")
        print("  Place files in data/raw/ as: semeval_train.conll, semeval_test.conll")
        print("  Or pass --train and --test arguments.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[INFO] Total tweets parsed: {len(combined):,}")

    # Save raw parsed version
    raw_out = os.path.join(RAW_DIR, "semeval2020_sentimix.csv")
    combined.to_csv(raw_out, index=False)
    print(f"[INFO] Raw parsed → {raw_out}")

    # Save pipeline-ready version
    pipeline_out = os.path.join(RAW_DIR, "semeval2020_sentimix_pipeline.csv")
    conll_to_pipeline_csv(combined, pipeline_out)

    print("\n[INFO] Done. Now run:")
    print("  python scripts/data_collection/02_explore_and_merge.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  type=str, default=None, help="Path to train CONLL file")
    parser.add_argument("--test",   type=str, default=None, help="Path to test CONLL file")
    parser.add_argument("--labels", type=str, default=None, help="Path to test labels file")
    args = parser.parse_args()
    main(args)
