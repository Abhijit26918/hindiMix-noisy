"""
Script 1: Download & collect source hate speech datasets.

Sources (in priority order):
  1. HASOC 2021 (Hindi-English code-mixed hate speech) — manual if HF fails
  2. LinCE / SemEval 2020 Task 9 Sentimix (Hindi-English code-mixed)
  3. Davidson 2017 (English hate speech)
  4. TweetEval-Hate (English hate tweets)
  5. UC Berkeley Measuring Hate Speech (~135K)
  6. OLID — Offensive Language Identification Dataset (~14K)
  7. HatEval SemEval 2019 Task 5 (~13K)
  8. Hindi offensive — community HF mirrors

Run: python scripts/data_collection/01_download_datasets.py
"""

import os
import zipfile
import io
import requests
import pandas as pd
from datasets import load_dataset

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def log(msg):
    print(f"[INFO] {msg}")


# ─────────────────────────────────────────────────────────────────
# 1. HASOC 2021 CodeMix — Hindi-English hate speech (primary source)
#    Source: github.com/AditiBagora/Hasoc2021CodeMix
# ─────────────────────────────────────────────────────────────────
def download_hasoc():
    log("Downloading HASOC 2021 CodeMix dataset...")

    # Check if already downloaded
    existing = [f for f in os.listdir(RAW_DIR) if "hasoc" in f.lower() and f.endswith(".csv")]
    if existing:
        log(f"  HASOC already downloaded: {existing} — skipping.")
        return

    GITHUB_ZIPS = {
        "train": "https://github.com/AditiBagora/Hasoc2021CodeMix/raw/main/Dataset/data-20220220T075606Z-001.zip",
        "test":  "https://github.com/AditiBagora/Hasoc2021CodeMix/raw/main/Dataset/test-20220220T075607Z-001.zip",
    }

    for split_name, url in GITHUB_ZIPS.items():
        try:
            log(f"  Downloading {split_name} zip from GitHub...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                log(f"  Zip contents: {z.namelist()}")
                for fname in z.namelist():
                    if fname.endswith((".csv", ".tsv", ".txt", ".xlsx")) and "__MACOSX" not in fname:
                        with z.open(fname) as zf:
                            try:
                                df = pd.read_csv(zf, sep=None, engine="python", encoding="utf-8")
                            except Exception:
                                zf.seek(0)
                                df = pd.read_csv(zf, sep="\t", encoding="utf-8")
                            out = f"{RAW_DIR}/hasoc2021_{split_name}.csv"
                            df.to_csv(out, index=False)
                            log(f"  Saved {len(df):,} rows ({split_name}) → hasoc2021_{split_name}.csv")
                            log(f"  Columns: {list(df.columns)}")
                            break
        except Exception as e:
            log(f"  GitHub download failed for {split_name}: {e}")

    # Fallback: HuggingFace mirrors
    existing = [f for f in os.listdir(RAW_DIR) if "hasoc" in f.lower() and f.endswith(".csv")]
    if not existing:
        log("  Trying HuggingFace mirrors...")
        for dataset_id in ["hasoc/hasoc2021", "victoriasovereigne/hasoc-2021"]:
            try:
                ds = load_dataset(dataset_id, trust_remote_code=False)
                for sname, sdata in ds.items():
                    df = pd.DataFrame(sdata)
                    df.to_csv(f"{RAW_DIR}/hasoc2021_{sname}.csv", index=False)
                    log(f"  Saved {len(df):,} ({sname}) → hasoc2021_{sname}.csv")
                return
            except Exception as e:
                log(f"  {dataset_id} failed: {e}")
        log("  [!] All auto-downloads failed. Manual download:")
        log("      https://hasocfire.github.io/hasoc/2021/dataset.html")
        log("      Save as: data/raw/hasoc2021_train.csv")


# ─────────────────────────────────────────────────────────────────
# 2. SemEval 2020 Task 9 Sentimix — Hindi-English code-mixed
#    Source: github.com/singhnivedita/SemEval2020-Task9 (pre-processed TSV)
#    Labels: 0=negative, 1=neutral, 2=positive (sentiment, not hate)
#    Used as code-mixed Hindi-English text source in the pipeline.
# ─────────────────────────────────────────────────────────────────
def download_semeval_sentimix():
    out = f"{RAW_DIR}/semeval2020_sentimix.csv"
    if os.path.exists(out):
        log("SemEval Sentimix already downloaded, skipping.")
        return
    log("Downloading SemEval 2020 Sentimix (Hindi-English code-mixed)...")

    BASE = "https://raw.githubusercontent.com/singhnivedita/SemEval2020-Task9/master/Fully%20Processed%20Datasets"
    files = {
        "train": f"{BASE}/FinalTrainingOnly.tsv",
        "val":   f"{BASE}/ValidationOnly.tsv",
        "test":  f"{BASE}/FinalTest.tsv",
    }

    dfs = []
    for split_name, url in files.items():
        try:
            df = pd.read_csv(url, sep="\t", header=None, names=["uid", "text", "label"])
            df["split"] = split_name
            dfs.append(df)
            log(f"  {split_name}: {len(df):,} samples")
        except Exception as e:
            log(f"  {split_name} failed: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(out, index=False)
        log(f"  Saved {len(combined):,} total samples → semeval2020_sentimix.csv")
    else:
        log("  All GitHub attempts failed.")
        log("  Manual: github.com/singhnivedita/SemEval2020-Task9 → Fully Processed Datasets/")


# ─────────────────────────────────────────────────────────────────
# 3. Davidson 2017 — English hate/offensive (already downloaded)
# ─────────────────────────────────────────────────────────────────
def download_davidson():
    out = f"{RAW_DIR}/davidson_hate_speech.csv"
    if os.path.exists(out):
        log(f"Davidson already downloaded ({out}), skipping.")
        return
    log("Downloading Davidson hate speech dataset...")
    try:
        ds = load_dataset("tdavidson/hate_speech_offensive", trust_remote_code=False)
        df = pd.DataFrame(ds["train"])
        df.to_csv(out, index=False)
        log(f"  Saved {len(df)} samples → davidson_hate_speech.csv")
    except Exception as e:
        log(f"  Davidson failed: {e}")


# ─────────────────────────────────────────────────────────────────
# 4. UC Berkeley Measuring Hate Speech — ~135K annotated comments
#    Best large-scale dataset: continuous score + binary hate label
# ─────────────────────────────────────────────────────────────────
def download_ucb_hate():
    out = f"{RAW_DIR}/ucb_measuring_hate_speech.csv"
    if os.path.exists(out):
        log(f"UCB Measuring Hate Speech already downloaded, skipping.")
        return
    log("Downloading UC Berkeley Measuring Hate Speech (~135K)...")
    try:
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", trust_remote_code=False)
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = pd.DataFrame(ds[split])
        df.to_csv(out, index=False)
        log(f"  Saved {len(df):,} samples → ucb_measuring_hate_speech.csv")
    except Exception as e:
        log(f"  UCB Measuring Hate Speech failed: {e}")


# ─────────────────────────────────────────────────────────────────
# 5. OLID — Offensive Language Identification Dataset (SemEval 2019)
# ─────────────────────────────────────────────────────────────────
def download_olid():
    out = f"{RAW_DIR}/olid.csv"
    if os.path.exists(out):
        log(f"OLID already downloaded, skipping.")
        return
    log("Downloading OLID dataset...")
    for dataset_id in ["christophsonntag/OLID", "olid", "tweeteval/olid"]:
        try:
            ds = load_dataset(dataset_id, trust_remote_code=False)
            dfs = []
            for split_name in ds.keys():
                df = pd.DataFrame(ds[split_name])
                df["split"] = split_name
                dfs.append(df)
            out_df = pd.concat(dfs, ignore_index=True)
            out_df.to_csv(out, index=False)
            log(f"  Saved {len(out_df):,} samples → olid.csv (via {dataset_id})")
            return
        except Exception as e:
            log(f"  {dataset_id} failed: {e}")
    log("  OLID not found — skipping.")


# ─────────────────────────────────────────────────────────────────
# 6. HatEval SemEval 2019 Task 5 — Hate vs immigrants & women
# ─────────────────────────────────────────────────────────────────
def download_hateval():
    out = f"{RAW_DIR}/hateval2019.csv"
    if os.path.exists(out):
        log(f"HatEval already downloaded, skipping.")
        return
    log("Downloading HatEval 2019 dataset...")
    for dataset_id in ["hateval2019", "HatEval", "SemEval2019Task5"]:
        try:
            ds = load_dataset(dataset_id, trust_remote_code=False)
            dfs = []
            for split_name in ds.keys():
                df = pd.DataFrame(ds[split_name])
                df["split"] = split_name
                dfs.append(df)
            out_df = pd.concat(dfs, ignore_index=True)
            out_df.to_csv(out, index=False)
            log(f"  Saved {len(out_df):,} samples → hateval2019.csv (via {dataset_id})")
            return
        except Exception as e:
            log(f"  {dataset_id} failed: {e}")
    log("  HatEval not found — skipping.")


# ─────────────────────────────────────────────────────────────────
# 5. TweetEval-Hate — English hate tweets (Twitter)
# ─────────────────────────────────────────────────────────────────
def download_tweeteval_hate():
    log("Downloading TweetEval-Hate dataset...")
    try:
        ds = load_dataset("tweet_eval", "hate", trust_remote_code=False)
        dfs = []
        for split_name in ds.keys():
            df = pd.DataFrame(ds[split_name])
            df["split"] = split_name
            dfs.append(df)
        out = pd.concat(dfs, ignore_index=True)
        out.to_csv(f"{RAW_DIR}/tweeteval_hate.csv", index=False)
        log(f"  Saved {len(out)} samples → tweeteval_hate.csv")
    except Exception as e:
        log(f"  TweetEval-Hate failed: {e}")


# ─────────────────────────────────────────────────────────────────
# 6. Hindi offensive — community HuggingFace mirrors
# ─────────────────────────────────────────────────────────────────
def download_hindi_offensive():
    log("Downloading Hindi offensive/hate speech dataset...")
    attempts = [
        ("Maha_Hate", None),
        ("mohitmayank/hindi_hate_speech_dataset", None),
        ("share_chat/hindi_hate_speech", None),
    ]
    for dataset_id, config in attempts:
        try:
            ds = load_dataset(dataset_id, config, trust_remote_code=False) if config else load_dataset(dataset_id, trust_remote_code=False)
            split = "train" if "train" in ds else list(ds.keys())[0]
            df = pd.DataFrame(ds[split])
            df.to_csv(f"{RAW_DIR}/hindi_offensive.csv", index=False)
            log(f"  Saved {len(df)} samples → hindi_offensive.csv (via {dataset_id})")
            return
        except Exception as e:
            log(f"  {dataset_id} failed: {e}")
    log("  Hindi offensive not found — skipping.")


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
def print_summary():
    log("\n=== Download Summary ===")
    found = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not found:
        log("  No files found in data/raw/")
        return
    total = 0
    for fname in sorted(found):
        path = os.path.join(RAW_DIR, fname)
        df = pd.read_csv(path)
        log(f"  {fname}: {len(df):,} rows | cols: {list(df.columns)}")
        total += len(df)
    log(f"\n  TOTAL: {total:,} rows across {len(found)} files")


if __name__ == "__main__":
    download_hasoc()
    download_semeval_sentimix()
    download_davidson()
    download_tweeteval_hate()
    download_ucb_hate()
    download_olid()
    download_hateval()
    download_hindi_offensive()
    print_summary()
    log("\nNext step: Run scripts/data_collection/02_explore_and_merge.py")
