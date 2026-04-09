"""
Classical Baselines: TF-IDF + SVM and TF-IDF + Logistic Regression.

Fastest baselines — no GPU needed. Good lower-bound comparison.

Run: python models/baselines/classical.py --noise_level clean
     python models/baselines/classical.py --noise_level medium
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

FINAL_DIR = "data/final"
RESULTS_DIR = "results/tables"
CHECKPOINT_DIR = "models/baselines/checkpoints/classical"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_data(noise_level):
    train_df = pd.read_csv(f"{FINAL_DIR}/train.csv").dropna(subset=["text", "label"])
    val_df = pd.read_csv(f"{FINAL_DIR}/val.csv").dropna(subset=["text", "label"])

    if noise_level != "all":
        train_df = train_df[train_df["noise_level"].isin(["clean", noise_level])]

    test_path = f"{FINAL_DIR}/test_clean.csv" if noise_level == "clean" else f"{FINAL_DIR}/test_noisy_{noise_level}.csv"
    if not os.path.exists(test_path):
        test_path = f"{FINAL_DIR}/test_clean.csv"
    test_df = pd.read_csv(test_path).dropna(subset=["text", "label"])

    return train_df, val_df, test_df


def build_pipeline(model_type):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # char n-grams — better for noisy/code-mixed text
        ngram_range=(2, 4),
        max_features=100_000,
        sublinear_tf=True,
    )
    if model_type == "svm":
        clf = LinearSVC(C=1.0, max_iter=2000)
    else:
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def evaluate(pipeline, X, y, split_name):
    preds = pipeline.predict(X)
    f1 = f1_score(y, preds, average="macro")
    print(f"\n{split_name} F1 (macro): {f1:.4f}")
    print(classification_report(y, preds, target_names=["Non-hate", "Hate"]))
    return f1, preds


def main(args):
    print(f"\n[Classical Baselines] noise_level={args.noise_level}")
    train_df, val_df, test_df = load_data(args.noise_level)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["label"].tolist()
    X_val = val_df["text"].astype(str).tolist()
    y_val = val_df["label"].tolist()
    X_test = test_df["text"].astype(str).tolist()
    y_test = test_df["label"].tolist()

    summary = []

    for model_type in ["svm", "lr"]:
        print(f"\n── TF-IDF + {model_type.upper()} ──")
        pipe = build_pipeline(model_type)
        pipe.fit(X_train, y_train)

        val_f1, _ = evaluate(pipe, X_val, y_val, "Val")
        test_f1, _ = evaluate(pipe, X_test, y_test, "Test")

        # Save model
        joblib.dump(pipe, f"{CHECKPOINT_DIR}/{model_type}_{args.noise_level}.joblib")

        result = {
            "model": f"tfidf_{model_type}",
            "noise_level": args.noise_level,
            "val_f1_macro": round(val_f1, 4),
            "test_f1_macro": round(test_f1, 4),
        }
        summary.append(result)

        out_path = f"{RESULTS_DIR}/tfidf_{model_type}_{args.noise_level}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved → {out_path}")

    print("\n── Summary ──")
    for r in summary:
        print(f"  {r['model']:12s} | val F1: {r['val_f1_macro']:.4f} | test F1: {r['test_f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_level", type=str, default="clean",
                        choices=["clean", "low", "medium", "high", "all"])
    args = parser.parse_args()
    main(args)
