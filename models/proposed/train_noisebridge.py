"""
Trainer for NoiseBridge — PWNIC contrastive + auxiliary noise prediction.

Usage:
    # ByT5 backbone (strongest, use RTX 4000)
    python models/proposed/train_noisebridge.py --encoder google/byt5-small --noise all

    # XLM-R backbone (faster, Kaggle/Colab)
    python models/proposed/train_noisebridge.py --encoder xlm-roberta-base --noise all

    # mBERT backbone
    python models/proposed/train_noisebridge.py --encoder bert-base-multilingual-cased --noise all

Arguments:
    --encoder   HuggingFace model name (default: google/byt5-small)
    --noise     clean | low | medium | high | all (default: all)
    --alpha     PWNIC loss weight (default: 0.5)
    --beta      auxiliary noise loss weight (default: 0.1)
    --epochs    (default: 5)
    --batch_size (default: 8, reduce to 4 for ByT5 on <16GB VRAM)
    --lr        (default: 3e-5)
    --fp16      use mixed precision (recommended for GPU)
"""

import os
import json
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from noisebridge import NoiseBridge, phonetic_weights

# ─────────────────────────────────────────────────────────────────
DATA_DIR    = "data/final"
RESULTS_DIR = "results/tables"
CKPT_DIR    = "models/proposed/checkpoints"
SEED        = 42

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED)
# ─────────────────────────────────────────────────────────────────

NOISE_LEVEL_MAP = {'clean': 0, 'low': 1, 'medium': 2, 'high': 3}


class NoiseBridgeDataset(Dataset):
    """
    Returns (clean_text, noisy_text, label, noise_level_id) triplets.
    Pairs clean and noisy versions of the same original text.
    """

    def __init__(self, clean_df: pd.DataFrame, noisy_df: pd.DataFrame,
                 tokenizer, max_len: int = 128):
        # Align on text_original
        merged = clean_df.merge(
            noisy_df[['text_original', 'text', 'noise_level']].rename(
                columns={'text': 'text_noisy', 'noise_level': 'noise_level_noisy'}
            ),
            on='text_original',
            how='inner'
        ).dropna(subset=['text', 'text_noisy', 'label'])

        self.clean_texts  = merged['text'].astype(str).tolist()
        self.noisy_texts  = merged['text_noisy'].astype(str).tolist()
        self.labels       = merged['label'].astype(int).tolist()
        self.noise_levels = [
            NOISE_LEVEL_MAP.get(str(n).lower(), 0)
            for n in merged['noise_level_noisy'].tolist()
        ]

        # Pre-compute phonetic weights (CPU, done once)
        print(f"  Computing phonetic weights for {len(self.clean_texts):,} pairs...")
        self.phi = phonetic_weights(self.clean_texts, self.noisy_texts)

        # Tokenize clean
        self.clean_enc = tokenizer(
            self.clean_texts, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        # Tokenize noisy
        self.noisy_enc = tokenizer(
            self.noisy_texts, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        print(f"  Dataset size: {len(self.labels):,}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':            self.clean_enc['input_ids'][idx],
            'attention_mask':       self.clean_enc['attention_mask'][idx],
            'noisy_input_ids':      self.noisy_enc['input_ids'][idx],
            'noisy_attention_mask': self.noisy_enc['attention_mask'][idx],
            'labels':               torch.tensor(self.labels[idx], dtype=torch.long),
            'noise_level_labels':   torch.tensor(self.noise_levels[idx], dtype=torch.long),
            'phi':                  self.phi[idx],
        }


class EvalDataset(Dataset):
    """Simple dataset for val/test — clean text only."""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.enc    = tokenizer(texts, max_length=max_len, padding='max_length',
                                truncation=True, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.enc['input_ids'][idx],
            'attention_mask': self.enc['attention_mask'][idx],
            'labels':         self.labels[idx],
        }


def collate_train(batch):
    return {
        'input_ids':            torch.stack([b['input_ids']            for b in batch]),
        'attention_mask':       torch.stack([b['attention_mask']       for b in batch]),
        'noisy_input_ids':      torch.stack([b['noisy_input_ids']      for b in batch]),
        'noisy_attention_mask': torch.stack([b['noisy_attention_mask'] for b in batch]),
        'labels':               torch.stack([b['labels']               for b in batch]),
        'noise_level_labels':   torch.stack([b['noise_level_labels']   for b in batch]),
        'phi':                  torch.stack([b['phi']                  for b in batch]),
    }


def collate_eval(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels':         torch.stack([b['labels']         for b in batch]),
    }


def load_data(noise_level):
    """Load and pair clean + noisy training data."""
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv").dropna(subset=['text', 'label'])
    val_df   = pd.read_csv(f"{DATA_DIR}/val.csv").dropna(subset=['text', 'label'])

    # Clean subset of train
    clean_train = train_df[train_df['noise_level'] == 'clean'].copy()

    # Noisy subset for this level
    if noise_level == 'all':
        noisy_train = train_df[train_df['noise_level'] != 'clean'].copy()
    else:
        noisy_train = train_df[train_df['noise_level'] == noise_level].copy()

    # Test set
    test_path = f"{DATA_DIR}/test_clean.csv" if noise_level == 'clean' \
                else f"{DATA_DIR}/test_noisy_{noise_level}.csv"
    if not os.path.exists(test_path):
        test_path = f"{DATA_DIR}/test_clean.csv"
    test_df = pd.read_csv(test_path).dropna(subset=['text', 'label'])

    print(f"  Clean train: {len(clean_train):,} | Noisy train: {len(noisy_train):,}")
    print(f"  Val: {len(val_df):,} | Test: {len(test_df):,}")
    return clean_train, noisy_train, val_df, test_df


def train_epoch(model, loader, optimizer, scheduler, device, class_weights,
                scaler=None, current_step=0, total_steps=1):
    model.train()
    total_loss = total_ce = total_pwnic = total_adv = total_aux = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        optimizer.zero_grad()

        # Anneal GRL lambda: 0 → lambda_max over training
        p = current_step / max(total_steps, 1)
        model.set_grl_lambda(p)
        current_step += 1

        kwargs = dict(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device),
            noisy_input_ids=batch['noisy_input_ids'].to(device),
            noisy_attention_mask=batch['noisy_attention_mask'].to(device),
            phi=batch['phi'].to(device),
            noise_level_labels=batch['noise_level_labels'].to(device),
            class_weights=class_weights,
        )

        if scaler:
            from torch.cuda.amp import autocast
            with autocast():
                out = model(**kwargs)
            scaler.scale(out['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(**kwargs)
            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        total_loss  += out['loss'].item()
        total_ce    += out['loss_ce'].item()    if out['loss_ce']    else 0
        total_pwnic += out['loss_pwnic'].item() if out['loss_pwnic'] else 0
        total_adv   += out['loss_adv'].item()   if out['loss_adv']   else 0
        total_aux   += out['loss_aux'].item()   if out['loss_aux']   else 0

    n = len(loader)
    return total_loss/n, total_ce/n, total_pwnic/n, total_adv/n, total_aux/n, current_step


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            )
            preds.extend(out['logits'].argmax(dim=-1).cpu().tolist())
            targets.extend(batch['labels'].tolist())
    return f1_score(targets, preds, average='macro'), preds, targets


def train_noise_level(args, noise_level, device):
    short_name = args.encoder.split('/')[-1].replace('-', '_')
    run_name   = f"noisebridge_{short_name}_{noise_level}"

    print(f"\n{'='*65}")
    print(f"NoiseBridge | encoder={args.encoder} | noise={noise_level}")
    print(f"α={args.alpha} (PWNIC) | β={args.beta} (aux) | fp16={args.fp16}")
    print(f"{'='*65}")

    clean_train, noisy_train, val_df, test_df = load_data(noise_level)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    max_len   = 128 if 'byt5' in args.encoder.lower() else 128

    # Class weights for 73/27 imbalance
    cw = compute_class_weight('balanced', classes=np.array([0, 1]),
                              y=clean_train['label'].values)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    print(f"  Class weights: [{cw[0]:.3f}, {cw[1]:.3f}]")

    train_ds = NoiseBridgeDataset(clean_train, noisy_train, tokenizer, max_len)
    val_ds   = EvalDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, max_len)
    test_ds  = EvalDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_train, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch_size * 2,
                              collate_fn=collate_eval, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size * 2,
                              collate_fn=collate_eval, num_workers=2)

    model = NoiseBridge(
        encoder_name=args.encoder,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lambda_max=args.lambda_max,
    ).to(device)

    # Gradient checkpointing for memory efficiency (ByT5)
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled.")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    scaler = None
    if args.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    ckpt_path    = f"{CKPT_DIR}/{run_name}.pt"
    best_val_f1  = 0.0
    total_steps  = len(train_loader) * args.epochs
    current_step = 0

    for epoch in range(args.epochs):
        loss, ce, pwnic, adv, aux, current_step = train_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights,
            scaler, current_step, total_steps
        )
        val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"loss:{loss:.4f} ce:{ce:.3f} pwnic:{pwnic:.3f} "
              f"adv:{adv:.3f} aux:{aux:.3f} | "
              f"val F1:{val_f1:.4f} | λ:{model.grl.current_lambda:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> best saved (val F1: {best_val_f1:.4f})")

    # Test on best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_f1, test_preds, test_targets = evaluate(model, test_loader, device)
    print(f"\n  Test F1 (macro): {test_f1:.4f}")
    print(classification_report(test_targets, test_preds, target_names=['Non-hate', 'Hate']))

    result = {
        'model':          run_name,
        'encoder':        args.encoder,
        'noise_level':    noise_level,
        'test_f1_macro':  round(test_f1, 4),
        'best_val_f1':    round(best_val_f1, 4),
        'alpha':          args.alpha,
        'beta':           args.beta,
        'gamma':          args.gamma,
        'lambda_max':     args.lambda_max,
        'epochs':         args.epochs,
        'lr':             args.lr,
        'batch_size':     args.batch_size,
    }
    out = f"{RESULTS_DIR}/{run_name}.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',    default='google/byt5-small')
    parser.add_argument('--noise',      default='all',
                        choices=['clean', 'low', 'medium', 'high', 'all'])
    parser.add_argument('--alpha',      type=float, default=0.15)
    parser.add_argument('--beta',       type=float, default=0.3)
    parser.add_argument('--gamma',      type=float, default=0.1)
    parser.add_argument('--lambda_max', type=float, default=1.0)
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=3e-5)
    parser.add_argument('--fp16',       action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    noise_levels = ['clean', 'low', 'medium', 'high'] if args.noise == 'all' else [args.noise]
    all_results  = []

    for noise in noise_levels:
        r = train_noise_level(args, noise, device)
        all_results.append(r)

    print(f"\n{'='*65}")
    print("NOISEBRIDGE SUMMARY")
    print(f"{'='*65}")
    print(f"{'noise':<10} {'val F1':>8} {'test F1':>9}")
    for r in all_results:
        print(f"  {r['noise_level']:<8} {r['best_val_f1']:>8.4f} {r['test_f1_macro']:>9.4f}")


if __name__ == '__main__':
    main()
