import pandas as pd, os

RAW = "data/raw"
files = {
    "davidson":  "davidson_hate_speech.csv",
    "olid":      "olid.csv",
    "semeval":   "semeval2020_sentimix.csv",
    "tweeteval": "tweeteval_hate.csv",
    "ucb":       "ucb_measuring_hate_speech.csv",
}

print("=== RAW DATA ===")
total = 0
for name, fname in files.items():
    path = os.path.join(RAW, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{name:12s}: {len(df):>8,} rows | cols: {list(df.columns[:5])}")
        total += len(df)
    else:
        print(f"{name:12s}: MISSING")
print(f"\nTOTAL RAW: {total:,} rows")

print("\n=== SEMEVAL SAMPLE (checking if Hindi-English) ===")
sem = pd.read_csv(os.path.join(RAW, "semeval2020_sentimix.csv"))
print(f"Columns: {list(sem.columns)}")
for i, row in sem.head(5).iterrows():
    print(f"  {list(row)[:3]}")

print("\n=== PROCESSED DATA ===")
proc_files = ["merged_clean.csv", "codemixed_clean.csv"]
for f in proc_files:
    path = f"data/processed/{f}"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{f}: {len(df):,} rows")
        if len(df) > 0:
            from collections import Counter
            print(f"  sources: {dict(Counter(df['source']))}")
            print(f"  lang_pairs: {dict(Counter(df['lang_pair']))}")
    else:
        print(f"{f}: MISSING")

print("\n=== FINAL SPLITS ===")
for f in ["train.csv","val.csv","test_clean.csv"]:
    path = f"data/final/{f}"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{f}: {len(df):,} rows")
    else:
        print(f"{f}: MISSING")
