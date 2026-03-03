"""
MuRIL Baseline Trainer for Hate Speech Detection.

MuRIL = Multilingual Representations for Indian Languages
Model: google/muril-base-cased (specifically trained on Indian languages!)
This is the STRONGEST baseline for Hindi-English code-mixed text.

Run: python models/baselines/muril_trainer.py --noise_level clean
     python models/baselines/muril_trainer.py --noise_level medium
"""

import os
import argparse
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

MODEL_NAME = "google/muril-base-cased"
FINAL_DIR = "data/final"
RESULTS_DIR = "results/tables"
CHECKPOINT_DIR = "models/baselines/checkpoints/muril"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Training"):
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating"):
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            )
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    f1 = f1_score(targets, preds, average="macro")
    return f1, preds, targets


# ─────────────────────────────────────────────────────────────────
def main(args):
    print(f"\n[MuRIL Baseline] Training on: {args.noise_level} | Device: {DEVICE}")

    # Load data
    train_df = pd.read_csv(f"{FINAL_DIR}/train.csv")
    val_df = pd.read_csv(f"{FINAL_DIR}/val.csv")

    # For noisy training: optionally filter by noise level
    if args.noise_level != "all":
        train_df = train_df[train_df["noise_level"].isin(["clean", args.noise_level])]

    test_path = f"{FINAL_DIR}/test_clean.csv" if args.noise_level == "clean" else f"{FINAL_DIR}/test_noisy_{args.noise_level}.csv"
    if not os.path.exists(test_path):
        test_path = f"{FINAL_DIR}/test_clean.csv"
    test_df = pd.read_csv(test_path)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    train_ds = HateSpeechDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_ds = HateSpeechDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)
    test_ds = HateSpeechDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    best_val_f1 = 0
    results = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_f1, _, _ = eval_epoch(model, val_loader)
        print(f"  Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(f"{CHECKPOINT_DIR}/best_{args.noise_level}")
            tokenizer.save_pretrained(f"{CHECKPOINT_DIR}/best_{args.noise_level}")
            print(f"  ✓ New best model saved (Val F1: {best_val_f1:.4f})")

        results.append({"epoch": epoch + 1, "train_loss": train_loss, "val_f1": val_f1})

    # Final test evaluation
    print("\n[INFO] Loading best model for test evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(f"{CHECKPOINT_DIR}/best_{args.noise_level}").to(DEVICE)
    test_f1, test_preds, test_targets = eval_epoch(model, test_loader)
    print(f"\nTest F1 (macro): {test_f1:.4f}")
    print(classification_report(test_targets, test_preds, target_names=["Non-hate", "Hate"]))

    # Save results
    result_summary = {
        "model": "MuRIL",
        "noise_level": args.noise_level,
        "test_f1_macro": test_f1,
        "best_val_f1": best_val_f1,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    out_path = f"{RESULTS_DIR}/muril_{args.noise_level}.json"
    with open(out_path, "w") as f:
        json.dump(result_summary, f, indent=2)
    print(f"\n[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_level", type=str, default="clean", choices=["clean", "low", "medium", "high", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    main(args)
