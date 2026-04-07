"""
ByT5 Baseline Trainer for Hate Speech Detection.

ByT5-small: byte-level T5, no tokenizer OOV — ideal for noisy Hinglish.
Uses T5ForSequenceClassification with gradient checkpointing for memory efficiency.

Run: python models/baselines/byt5_trainer.py --noise_level clean
     python models/baselines/byt5_trainer.py --noise_level all --fp16
"""

import os
import argparse
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    T5ForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

MODEL_NAME     = "google/byt5-small"
FINAL_DIR      = "data/final"
RESULTS_DIR    = "results/tables"
CHECKPOINT_DIR = "models/baselines/checkpoints/byt5"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc    = tokenizer(
            texts, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.enc['input_ids'][idx],
            'attention_mask': self.enc['attention_mask'][idx],
            'labels':         self.labels[idx],
        }


def train_epoch(model, loader, optimizer, scheduler, scaler=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="  Training"):
        optimizer.zero_grad()
        kwargs = dict(
            input_ids=batch['input_ids'].to(DEVICE),
            attention_mask=batch['attention_mask'].to(DEVICE),
            labels=batch['labels'].to(DEVICE),
        )
        if scaler:
            from torch.amp import autocast
            with autocast('cuda'):
                out = model(**kwargs)
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(**kwargs)
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
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
            )
            preds.extend(out.logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch['labels'].tolist())
    return f1_score(targets, preds, average='macro'), preds, targets


def main(args):
    print(f"\n[ByT5 Baseline] noise_level={args.noise_level} | device={DEVICE} | fp16={args.fp16}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_df = pd.read_csv(f"{FINAL_DIR}/train.csv").dropna(subset=['text', 'label'])
    val_df   = pd.read_csv(f"{FINAL_DIR}/val.csv").dropna(subset=['text', 'label'])

    if args.noise_level != 'all':
        train_df = train_df[train_df['noise_level'].isin(['clean', args.noise_level])]

    test_path = (f"{FINAL_DIR}/test_clean.csv" if args.noise_level == 'clean'
                 else f"{FINAL_DIR}/test_noisy_{args.noise_level}.csv")
    if not os.path.exists(test_path):
        test_path = f"{FINAL_DIR}/test_clean.csv"
    test_df = pd.read_csv(test_path).dropna(subset=['text', 'label'])

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    # Memory efficiency — essential for ByT5 on long byte sequences
    model.gradient_checkpointing_enable()

    train_ds = HateSpeechDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer)
    val_ds   = HateSpeechDataset(val_df['text'].tolist(),   val_df['label'].tolist(),   tokenizer)
    test_ds  = HateSpeechDataset(test_df['text'].tolist(),  test_df['label'].tolist(),  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch_size * 2, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size * 2, num_workers=2)

    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    scaler = None
    if args.fp16:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda')

    best_val_f1  = 0
    ckpt_path    = f"{CHECKPOINT_DIR}/best_{args.noise_level}"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        loss  = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_f1, _, _ = eval_epoch(model, val_loader)
        print(f"  Loss: {loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  -> best saved (Val F1: {best_val_f1:.4f})")

    print("\n[INFO] Loading best model for test evaluation...")
    model = T5ForSequenceClassification.from_pretrained(ckpt_path, num_labels=2).to(DEVICE)
    test_f1, test_preds, test_targets = eval_epoch(model, test_loader)
    print(f"\nTest F1 (macro): {test_f1:.4f}")
    print(classification_report(test_targets, test_preds, target_names=['Non-hate', 'Hate']))

    result = {
        'model':          'ByT5',
        'noise_level':    args.noise_level,
        'test_f1_macro':  round(test_f1, 4),
        'best_val_f1':    round(best_val_f1, 4),
        'epochs':         args.epochs,
        'lr':             args.lr,
        'batch_size':     args.batch_size,
    }
    out_path = f"{RESULTS_DIR}/byt5_{args.noise_level}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Results saved -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', default='all',
                        choices=['clean', 'low', 'medium', 'high', 'all'])
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=3e-5)
    parser.add_argument('--fp16',       action='store_true')
    args = parser.parse_args()
    main(args)
