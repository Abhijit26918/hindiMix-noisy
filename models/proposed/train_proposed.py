"""
Trainer for NoiseRobustHateDetector (ByT5 + phonetic + noise-aware attention).

Usage:
    python models/proposed/train_proposed.py --noise clean
    python models/proposed/train_proposed.py --noise low
    python models/proposed/train_proposed.py --noise medium
    python models/proposed/train_proposed.py --noise high
    python models/proposed/train_proposed.py --noise all   # runs all 4

Requirements:
    pip install transformers torch jellyfish scikit-learn pandas tqdm
"""

import os
import json
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# Import model from same directory
import sys
sys.path.insert(0, os.path.dirname(__file__))
from noise_robust_model import NoiseRobustHateDetector

# ─────────────────────────────────────────────────────────────────
DATA_DIR    = "data/final"
RESULTS_DIR = "results/tables"
CKPT_DIR    = "models/proposed/checkpoints"
MODEL_NAME  = "google/byt5-small"

EPOCHS      = 5
BATCH_SIZE  = 16        # reduce to 8 if OOM on 8GB VRAM
MAX_LEN     = 256       # bytes; ByT5 is byte-level so ~= chars
LR          = 3e-5
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED)
# ─────────────────────────────────────────────────────────────────


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.texts      = texts          # kept for phonetic features
        self.tokenizer  = tokenizer
        self.max_len    = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
            "text":           self.texts[idx],
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
        "texts":          [b["text"] for b in batch],
    }


def load_data(noise_level):
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv").dropna(subset=["text", "label"])
    val_df   = pd.read_csv(f"{DATA_DIR}/val.csv").dropna(subset=["text", "label"])

    if noise_level != "all":
        train_df = train_df[train_df["noise_level"].isin(["clean", noise_level])]

    test_path = (
        f"{DATA_DIR}/test_clean.csv" if noise_level == "clean"
        else f"{DATA_DIR}/test_noisy_{noise_level}.csv"
    )
    if not os.path.exists(test_path):
        test_path = f"{DATA_DIR}/test_clean.csv"
    test_df = pd.read_csv(test_path).dropna(subset=["text", "label"])

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  train", leave=False):
        optimizer.zero_grad()
        # Tokenize per-batch word tokens for phonetic features
        token_strings = [text.split() for text in batch["texts"]]
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_strings=token_strings,
            labels=batch["labels"].to(device),
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out["loss"].item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            token_strings = [text.split() for text in batch["texts"]]
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_strings=token_strings,
            )
            preds.extend(out["logits"].argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].tolist())
    f1 = f1_score(targets, preds, average="macro")
    return f1, preds, targets


def train_noise_level(noise_level, device):
    print(f"\n{'='*60}")
    print(f"NoiseRobustHateDetector | noise={noise_level} | device={device}")
    print(f"{'='*60}")

    train_df, val_df, test_df = load_data(noise_level)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = NoiseRobustHateDetector(num_labels=2).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = HateSpeechDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    val_ds   = HateSpeechDataset(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer)
    test_ds  = HateSpeechDataset(test_df["text"].tolist(),  test_df["label"].tolist(),  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    ckpt_path = f"{CKPT_DIR}/byt5_{noise_level}.pt"
    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{EPOCHS} | loss: {loss:.4f} | val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> best saved ({best_val_f1:.4f})")

    # Load best and test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_f1, test_preds, test_targets = evaluate(model, test_loader, device)
    print(f"\n  Test F1 (macro): {test_f1:.4f}")
    print(classification_report(test_targets, test_preds, target_names=["Non-hate", "Hate"]))

    result = {
        "model": "byt5_proposed",
        "noise_level": noise_level,
        "test_f1_macro": round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
    }
    out = f"{RESULTS_DIR}/byt5_proposed_{noise_level}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", default="all",
                        choices=["clean", "low", "medium", "high", "all"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--lr",         type=float, default=LR)
    args = parser.parse_args()

    # Allow CLI overrides
    global BATCH_SIZE, EPOCHS, LR
    BATCH_SIZE = args.batch_size
    EPOCHS     = args.epochs
    LR         = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    noise_levels = ["clean", "low", "medium", "high"] if args.noise == "all" else [args.noise]
    all_results = []

    for noise in noise_levels:
        r = train_noise_level(noise, device)
        all_results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'noise':<10} {'val F1':>8} {'test F1':>9}")
    for r in all_results:
        print(f"  {r['noise_level']:<8} {r['best_val_f1']:>8.4f} {r['test_f1_macro']:>9.4f}")


if __name__ == "__main__":
    main()
