"""
Trainer for NoiseBridge — corrected version.

Fixes vs v1 (the version that produced under-baseline results):

  Bug 1 — Half the training data:
    v1 inner-merged clean and noisy on text_original and trained ONLY on
    paired examples, with CE only on z_clean. Per-epoch CE supervision was
    ~half the baseline. Fixed by the per-row anchor scheme: each baseline
    row → exactly one item here, with a per-row CE mask (ce_mask_clean or
    ce_mask_noisy). Total CE updates per epoch = N_clean + N_noisy ✓

  Bug 2 — Degenerate noise labels in per-noise-level runs:
    When training on e.g. --noise high, every noisy row has noise_level=3,
    so the adversarial / aux head was trained against a CONSTANT target.
    The adversarial head with GRL at λ→1 then pushed the encoder to make
    z_noisy noise-shaped, which collapsed XLM-R high to 0.7773. Fixed by:
      * Removing the adversarial head entirely (see noisebridge.py).
      * Auto-disabling the aux head when the dataset has fewer than 2
        distinct noise levels (`enable_aux` flag).

  Bug 3 — adv and aux fighting on the same z_noisy:
    Both heads consumed z_noisy with opposite gradient signs, and β=0.3
    swamped γ=0.1 so aux became gradient noise. Fixed by removing the
    adversarial head — the paper now tells a single, clean
    "PWNIC + multi-task noise prediction" story.

  Bug 4 — `--noise clean` was degenerate:
    v1 set noisy_train = clean_train when noise_level=='clean', so the
    "noisy" half of every pair was identical to the clean half (phi=0,
    aux targets all 0). Fixed: clean condition skips pair construction
    entirely → vanilla classifier (the only honest interpretation).

  Bug 5 — Reproducibility:
    v1 only seeded torch CPU. Now seeding random + numpy + torch CPU + CUDA,
    cudnn.deterministic, and DataLoader workers via worker_init_fn +
    torch.Generator. Multi-seed runs are supported via --seeds.

  Bug 6 — ByT5 truncation:
    v1 used max_len=128 for ByT5 (byte-level). Hindi UTF-8 is 3 bytes/char,
    so 128 bytes ≈ 40 Devanagari characters. Default raised to 256 for ByT5,
    overridable via --max_len.

Usage:
    # Single seed, all noise levels (matches v1 setup, fast comparison)
    python train_noisebridge.py --encoder google/byt5-small --noise all --fp16

    # Multi-seed sanity run on the most informative cells
    python train_noisebridge.py --encoder bert-base-multilingual-cased \\
        --noise all --seeds 42 43 44 --fp16

    # Just rerun the cell that collapsed in v1
    python train_noisebridge.py --encoder xlm-roberta-base --noise high --fp16
"""

import os
import json
import random
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noisebridge import NoiseBridge, phonetic_weights

# ─────────────────────────────────────────────────────────────────
DATA_DIR    = "data/final"
RESULTS_DIR = "results/tables"
CKPT_DIR    = "models/proposed/checkpoints"
# ─────────────────────────────────────────────────────────────────

NOISE_LEVEL_MAP = {'clean': 0, 'low': 1, 'medium': 2, 'high': 3}


# ─────────────────────────────────────────────────────────────────
# Reproducibility helpers
# ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ─────────────────────────────────────────────────────────────────
# Dataset — per-row anchor scheme
# ─────────────────────────────────────────────────────────────────

class NoiseBridgeDataset(Dataset):
    """
    Each item corresponds to exactly one row in the baseline trainer's
    training set:

      anchor type            ce_mask_clean   ce_mask_noisy   has_pair (if partner exists)
      ─────────────────────  ─────────────   ─────────────   ────────
      clean row, has noisy        True           False             True
      clean row, no noisy         True           False             False
      noisy row, has clean        False          True              True
      noisy row, no clean         False          True              False

    Total items = N_clean + N_noisy_in_condition (matches baseline exactly).
    Each item contributes exactly one CE update per epoch.

    Pairs additionally contribute PWNIC and (if enable_aux) noise-prediction
    losses.
    """

    def __init__(self, train_df: pd.DataFrame, noise_level: str,
                 tokenizer, max_len: int = 256):
        if 'text_original' not in train_df.columns:
            raise ValueError(
                "train.csv must contain a 'text_original' column to pair "
                "clean and noisy versions of the same utterance."
            )

        train_df = train_df.dropna(subset=['text', 'label']).copy()
        train_df['text_original'] = train_df['text_original'].astype(str)

        clean_df = train_df[train_df['noise_level'] == 'clean']

        if noise_level == 'clean':
            noisy_df = train_df.iloc[0:0]
        elif noise_level == 'all':
            noisy_df = train_df[train_df['noise_level'].isin(['low', 'medium', 'high'])]
        else:
            noisy_df = train_df[train_df['noise_level'] == noise_level]

        # Anchor df = baseline's training rows
        if noise_level == 'clean':
            anchor_df = clean_df
        else:
            anchor_df = pd.concat([clean_df, noisy_df], ignore_index=True)

        # Lookups for finding partners
        clean_text_by_orig = dict(zip(
            clean_df['text_original'], clean_df['text'].astype(str)
        ))

        # For each text_original, the FIRST noisy version in document order
        # (deterministic; clean anchors get a single canonical noisy partner)
        noisy_lookup = {}
        for _, r in noisy_df.iterrows():
            orig = r['text_original']
            if orig not in noisy_lookup:
                noisy_lookup[orig] = (str(r['text']), str(r['noise_level']).lower())

        clean_texts     = []
        noisy_texts     = []
        labels          = []
        noise_ids       = []
        ce_mask_clean   = []
        ce_mask_noisy   = []
        has_pair        = []

        for _, row in anchor_df.iterrows():
            anchor_text  = str(row['text'])
            anchor_label = int(row['label'])
            anchor_level = str(row['noise_level']).lower()
            text_orig    = row['text_original']

            if anchor_level == 'clean':
                # Anchor is a clean row → CE on clean. Look for a noisy partner.
                partner = noisy_lookup.get(text_orig)
                if partner is not None:
                    p_text, p_level = partner
                    clean_texts.append(anchor_text)
                    noisy_texts.append(p_text)
                    has_pair.append(True)
                    noise_ids.append(NOISE_LEVEL_MAP[p_level])
                else:
                    clean_texts.append(anchor_text)
                    noisy_texts.append(anchor_text)   # placeholder
                    has_pair.append(False)
                    noise_ids.append(0)
                ce_mask_clean.append(True)
                ce_mask_noisy.append(False)
            else:
                # Anchor is a noisy row → CE on noisy. Look for clean partner.
                clean_partner = clean_text_by_orig.get(text_orig)
                if clean_partner is not None:
                    clean_texts.append(clean_partner)
                    noisy_texts.append(anchor_text)
                    has_pair.append(True)
                    noise_ids.append(NOISE_LEVEL_MAP[anchor_level])
                else:
                    clean_texts.append(anchor_text)   # placeholder
                    noisy_texts.append(anchor_text)
                    has_pair.append(False)
                    noise_ids.append(NOISE_LEVEL_MAP[anchor_level])
                ce_mask_clean.append(False)
                ce_mask_noisy.append(True)

            labels.append(anchor_label)

        if len(clean_texts) == 0:
            raise RuntimeError(f"No training examples assembled for noise_level={noise_level}")

        # ── Tokenize ──
        print(f"  Tokenizing {len(clean_texts):,} examples (max_len={max_len})...")
        self.clean_enc = tokenizer(
            clean_texts, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        self.noisy_enc = tokenizer(
            noisy_texts, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        # ── Phonetic weights ──
        print(f"  Computing phonetic weights...")
        self.phi = phonetic_weights(clean_texts, noisy_texts)

        self.labels        = torch.tensor(labels,        dtype=torch.long)
        self.noise_ids     = torch.tensor(noise_ids,     dtype=torch.long)
        self.ce_mask_clean = torch.tensor(ce_mask_clean, dtype=torch.bool)
        self.ce_mask_noisy = torch.tensor(ce_mask_noisy, dtype=torch.bool)
        self.has_pair      = torch.tensor(has_pair,      dtype=torch.bool)

        n_pair        = int(self.has_pair.sum().item())
        n_total       = len(self.labels)
        n_solo        = n_total - n_pair
        ce_per_epoch  = n_total   # exactly one CE per row

        # Auto-disable aux if only one noise level is present in pairs.
        unique_noise = set()
        if n_pair > 0:
            paired_levels = self.noise_ids[self.has_pair].tolist()
            unique_noise  = set(paired_levels)
        self.has_diverse_noise = len(unique_noise) >= 2

        print(f"  Dataset: {n_pair:,} paired + {n_solo:,} solo = {n_total:,} items")
        print(f"  CE supervisions / epoch: {ce_per_epoch:,}  (matches baseline)")
        print(f"  Distinct noise levels in pairs: {sorted(unique_noise)} "
              f"→ aux head: {'enabled' if self.has_diverse_noise else 'DISABLED'}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'clean_input_ids':       self.clean_enc['input_ids'][idx],
            'clean_attention_mask':  self.clean_enc['attention_mask'][idx],
            'noisy_input_ids':       self.noisy_enc['input_ids'][idx],
            'noisy_attention_mask':  self.noisy_enc['attention_mask'][idx],
            'labels':                self.labels[idx],
            'noise_level_labels':    self.noise_ids[idx],
            'ce_mask_clean':         self.ce_mask_clean[idx],
            'ce_mask_noisy':         self.ce_mask_noisy[idx],
            'has_pair':              self.has_pair[idx],
            'phi':                   self.phi[idx],
        }


class EvalDataset(Dataset):
    """Simple dataset for val / test — text only."""

    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.enc = tokenizer(
            texts, max_length=max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.enc['input_ids'][idx],
            'attention_mask': self.enc['attention_mask'][idx],
            'labels':         self.labels[idx],
        }


def collate_train(batch):
    return {
        'clean_input_ids':       torch.stack([b['clean_input_ids']       for b in batch]),
        'clean_attention_mask':  torch.stack([b['clean_attention_mask']  for b in batch]),
        'noisy_input_ids':       torch.stack([b['noisy_input_ids']       for b in batch]),
        'noisy_attention_mask':  torch.stack([b['noisy_attention_mask']  for b in batch]),
        'labels':                torch.stack([b['labels']                for b in batch]),
        'noise_level_labels':    torch.stack([b['noise_level_labels']    for b in batch]),
        'ce_mask_clean':         torch.stack([b['ce_mask_clean']         for b in batch]),
        'ce_mask_noisy':         torch.stack([b['ce_mask_noisy']         for b in batch]),
        'has_pair':              torch.stack([b['has_pair']              for b in batch]),
        'phi':                   torch.stack([b['phi']                   for b in batch]),
    }


def collate_eval(batch):
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels':         torch.stack([b['labels']         for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_data(noise_level):
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv").dropna(subset=['text', 'label'])
    val_df   = pd.read_csv(f"{DATA_DIR}/val.csv").dropna(subset=['text', 'label'])

    if noise_level in ('clean', 'all'):
        test_path = f"{DATA_DIR}/test_clean.csv"
    else:
        test_path = f"{DATA_DIR}/test_noisy_{noise_level}.csv"
    if not os.path.exists(test_path):
        print(f"  [warn] {test_path} not found, falling back to test_clean.csv")
        test_path = f"{DATA_DIR}/test_clean.csv"

    test_df = pd.read_csv(test_path).dropna(subset=['text', 'label'])
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, device, class_weights, scaler=None):
    model.train()
    totals = {'loss': 0.0, 'ce': 0.0, 'pwnic': 0.0, 'aux': 0.0}

    for batch in tqdm(loader, desc="  train", leave=False):
        optimizer.zero_grad()

        kwargs = dict(
            clean_input_ids       = batch['clean_input_ids'].to(device),
            clean_attention_mask  = batch['clean_attention_mask'].to(device),
            noisy_input_ids       = batch['noisy_input_ids'].to(device),
            noisy_attention_mask  = batch['noisy_attention_mask'].to(device),
            labels                = batch['labels'].to(device),
            ce_mask_clean         = batch['ce_mask_clean'].to(device),
            ce_mask_noisy         = batch['ce_mask_noisy'].to(device),
            has_pair              = batch['has_pair'].to(device),
            phi                   = batch['phi'].to(device),
            noise_level_labels    = batch['noise_level_labels'].to(device),
            class_weights         = class_weights,
        )

        if scaler is not None:
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

        totals['loss']  += out['loss'].item()
        totals['ce']    += out['loss_ce'].item()
        totals['pwnic'] += out['loss_pwnic'].item()
        totals['aux']   += out['loss_aux'].item()

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            logits = model.predict(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
            )
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch['labels'].tolist())
    return f1_score(targets, preds, average='macro'), preds, targets


# ─────────────────────────────────────────────────────────────────
# One full training run (one encoder × one noise condition × one seed)
# ─────────────────────────────────────────────────────────────────

def train_one(args, noise_level, device, seed):
    set_seed(seed)

    short_name = args.encoder.split('/')[-1].replace('-', '_')
    run_name   = f"noisebridge_{short_name}_{noise_level}_s{seed}"

    print(f"\n{'='*72}")
    print(f"NoiseBridge | encoder={args.encoder} | noise={noise_level} | seed={seed}")
    print(f"alpha={args.alpha} (PWNIC) | gamma={args.gamma} (aux) | fp16={args.fp16}")
    print(f"{'='*72}")

    train_df, val_df, test_df = load_data(noise_level)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    if args.max_len is not None:
        max_len = args.max_len
    elif 'byt5' in args.encoder.lower():
        max_len = 256        # byte-level: Hindi UTF-8 is 3 bytes/char
    else:
        max_len = 128

    train_ds = NoiseBridgeDataset(train_df, noise_level, tokenizer, max_len=max_len)
    val_ds   = EvalDataset(val_df['text'].tolist(),  val_df['label'].tolist(),
                           tokenizer, max_len=max_len)
    test_ds  = EvalDataset(test_df['text'].tolist(), test_df['label'].tolist(),
                           tokenizer, max_len=max_len)

    print(f"  Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # Class weights from the actual labels NoiseBridge will see during training.
    cw = compute_class_weight('balanced', classes=np.array([0, 1]),
                              y=train_ds.labels.numpy())
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    print(f"  Class weights: [{cw[0]:.3f}, {cw[1]:.3f}]")

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_train, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        collate_fn=collate_eval, num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2,
        collate_fn=collate_eval, num_workers=args.num_workers,
        worker_init_fn=seed_worker,
    )

    enable_aux = train_ds.has_diverse_noise

    model = NoiseBridge(
        encoder_name = args.encoder,
        alpha        = args.alpha,
        gamma        = args.gamma,
        enable_aux   = enable_aux,
    ).to(device)

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

    ckpt_path   = f"{CKPT_DIR}/{run_name}.pt"
    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        stats        = train_epoch(model, train_loader, optimizer, scheduler,
                                   device, class_weights, scaler)
        val_f1, _, _ = evaluate(model, val_loader, device)

        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"loss:{stats['loss']:.4f} ce:{stats['ce']:.3f} "
              f"pwnic:{stats['pwnic']:.3f} aux:{stats['aux']:.3f} | "
              f"val F1:{val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> best saved (val F1: {best_val_f1:.4f})")

    # Test on best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_f1, test_preds, test_targets = evaluate(model, test_loader, device)
    print(f"\n  Test F1 (macro): {test_f1:.4f}")
    print(classification_report(test_targets, test_preds,
                                target_names=['Non-hate', 'Hate']))

    result = {
        'model':         run_name,
        'encoder':       args.encoder,
        'noise_level':   noise_level,
        'seed':          seed,
        'test_f1_macro': round(test_f1, 4),
        'best_val_f1':   round(best_val_f1, 4),
        'alpha':         args.alpha,
        'gamma':         args.gamma,
        'enable_aux':    enable_aux,
        'epochs':        args.epochs,
        'lr':            args.lr,
        'batch_size':    args.batch_size,
        'max_len':       max_len,
    }
    out_path = f"{RESULTS_DIR}/{run_name}.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}")
    return result


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',     default='google/byt5-small')
    parser.add_argument('--noise',       default='all',
                        choices=['clean', 'low', 'medium', 'high', 'all'])
    parser.add_argument('--alpha',       type=float, default=0.15,
                        help='PWNIC contrastive loss weight')
    parser.add_argument('--gamma',       type=float, default=0.2,
                        help='Auxiliary noise prediction loss weight')
    parser.add_argument('--epochs',      type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=3e-5)
    parser.add_argument('--max_len',     type=int,   default=None,
                        help='Override default max_len (byt5: 256, others: 128)')
    parser.add_argument('--num_workers', type=int,   default=2)
    parser.add_argument('--seeds',       type=int,   nargs='+', default=[42],
                        help='Seeds for multi-seed runs (e.g. --seeds 42 43 44)')
    parser.add_argument('--fp16',        action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU:  {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    noise_levels = ['clean', 'low', 'medium', 'high'] if args.noise == 'all' else [args.noise]
    all_results  = []

    for noise in noise_levels:
        for seed in args.seeds:
            r = train_one(args, noise, device, seed)
            all_results.append(r)

    # Per-run summary
    print(f"\n{'='*72}")
    print("NOISEBRIDGE SUMMARY")
    print(f"{'='*72}")
    print(f"{'noise':<10} {'seed':>5} {'val F1':>9} {'test F1':>9}")
    for r in all_results:
        print(f"  {r['noise_level']:<8} {r['seed']:>5} "
              f"{r['best_val_f1']:>9.4f} {r['test_f1_macro']:>9.4f}")

    # Mean ± std across seeds (only meaningful with >1 seed)
    if len(args.seeds) > 1:
        print(f"\n{'='*72}")
        print("MEAN ± STD ACROSS SEEDS")
        print(f"{'='*72}")
        df  = pd.DataFrame(all_results)
        agg = df.groupby('noise_level')['test_f1_macro'].agg(['mean', 'std']).reset_index()
        for _, row in agg.iterrows():
            std = row['std'] if not pd.isna(row['std']) else 0.0
            print(f"  {row['noise_level']:<10} {row['mean']:.4f} ± {std:.4f}")


if __name__ == '__main__':
    main()
