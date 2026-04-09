"""
Character-level CNN Baseline for Hate Speech Detection.

Operates at character level — naturally robust to spelling noise and
transliteration variation. No pretrained weights needed.

Architecture:
  Char embedding → Conv1D (multiple filter sizes) → MaxPool → FC → Softmax

Run: python models/baselines/char_cnn.py --noise_level clean
     python models/baselines/char_cnn.py --noise_level medium
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

FINAL_DIR = "data/final"
RESULTS_DIR = "results/tables"
CHECKPOINT_DIR = "models/baselines/checkpoints/char_cnn"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Character vocabulary: printable ASCII + common Hindi romanization chars
CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
             ".,!?@#&'-_:/\"\n\t") + ["<PAD>", "<UNK>"]
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
PAD_IDX = CHAR2IDX["<PAD>"]
UNK_IDX = CHAR2IDX["<UNK>"]
VOCAB_SIZE = len(CHARS)


def text_to_ids(text, max_len=300):
    ids = [CHAR2IDX.get(c, UNK_IDX) for c in text[:max_len]]
    ids += [PAD_IDX] * (max_len - len(ids))
    return ids


# ─────────────────────────────────────────────────────────────────
class CharDataset(Dataset):
    def __init__(self, texts, labels, max_len=300):
        self.data = [torch.tensor(text_to_ids(t, max_len), dtype=torch.long) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], torch.tensor(self.labels[idx], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────
class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128,
                 filter_sizes=(3, 4, 5), num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)   # (batch, embed_dim, seq_len)
        pooled = [F.max_pool1d(F.relu(conv(emb)), conv(emb).size(2)).squeeze(2)
                  for conv in self.convs]
        out = torch.cat(pooled, dim=1)              # (batch, num_filters * len(filter_sizes))
        out = self.dropout(out)
        return self.fc(out)


# ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="  Training"):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="  Evaluating"):
            logits = model(X.to(DEVICE))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(y.tolist())
    f1 = f1_score(targets, preds, average="macro")
    return f1, preds, targets


# ─────────────────────────────────────────────────────────────────
def main(args):
    print(f"\n[Char-CNN Baseline] noise_level={args.noise_level} | device={DEVICE}")

    train_df = pd.read_csv(f"{FINAL_DIR}/train.csv").dropna(subset=["text", "label"])
    val_df = pd.read_csv(f"{FINAL_DIR}/val.csv").dropna(subset=["text", "label"])

    if args.noise_level != "all":
        train_df = train_df[train_df["noise_level"].isin(["clean", args.noise_level])]

    test_path = f"{FINAL_DIR}/test_clean.csv" if args.noise_level == "clean" else f"{FINAL_DIR}/test_noisy_{args.noise_level}.csv"
    if not os.path.exists(test_path):
        test_path = f"{FINAL_DIR}/test_clean.csv"
    test_df = pd.read_csv(test_path).dropna(subset=["text", "label"])

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = CharDataset(train_df["text"].astype(str).tolist(), train_df["label"].tolist())
    val_ds = CharDataset(val_df["text"].astype(str).tolist(), val_df["label"].tolist())
    test_ds = CharDataset(test_df["text"].astype(str).tolist(), test_df["label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=2)

    model = CharCNN(vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_f1, _, _ = eval_epoch(model, val_loader)
        print(f"  Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_{args.noise_level}.pt")
            print(f"  ✓ Best saved (Val F1: {best_val_f1:.4f})")

    # Test evaluation
    print("\n[INFO] Loading best model for test evaluation...")
    model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/best_{args.noise_level}.pt", map_location=DEVICE))
    test_f1, test_preds, test_targets = eval_epoch(model, test_loader)
    print(f"\nTest F1 (macro): {test_f1:.4f}")
    print(classification_report(test_targets, test_preds, target_names=["Non-hate", "Hate"]))

    result = {
        "model": "CharCNN",
        "noise_level": args.noise_level,
        "test_f1_macro": round(test_f1, 4),
        "best_val_f1": round(best_val_f1, 4),
        "epochs": args.epochs,
        "lr": args.lr,
    }
    out_path = f"{RESULTS_DIR}/char_cnn_{args.noise_level}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_level", type=str, default="clean",
                        choices=["clean", "low", "medium", "high", "all"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
