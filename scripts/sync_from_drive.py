"""
Sync results from Google Drive back to local project.

Two modes:
  1. Auto  — if Google Drive desktop app is installed, copies directly from Drive folder
  2. Manual — point it at any folder/zip you downloaded from Colab

Usage:
  python scripts/sync_from_drive.py
  python scripts/sync_from_drive.py --drive_path "C:/Users/abhij/Google Drive/hindiMix-noisy"
  python scripts/sync_from_drive.py --zip path/to/results_tables.zip
"""

import os
import sys
import shutil
import argparse
import zipfile

LOCAL_PROJECT = os.path.abspath(".")
RESULTS_DIR   = os.path.join(LOCAL_PROJECT, "results", "tables")
FIGURES_DIR   = os.path.join(LOCAL_PROJECT, "results", "figures")

# Common Google Drive paths on Windows
DRIVE_CANDIDATES = [
    os.path.expanduser("~/Google Drive/hindiMix-noisy"),
    os.path.expanduser("~/OneDrive/hindiMix-noisy"),
    "G:/My Drive/hindiMix-noisy",
    "G:/MyDrive/hindiMix-noisy",
    "D:/Google Drive/hindiMix-noisy",
]


def log(msg):
    print(f"[SYNC] {msg}")


def sync_from_folder(drive_path):
    log(f"Syncing from: {drive_path}")

    mappings = [
        (os.path.join(drive_path, "results", "tables"),  RESULTS_DIR),
        (os.path.join(drive_path, "results", "figures"), FIGURES_DIR),
    ]

    total = 0
    for src_dir, dst_dir in mappings:
        if not os.path.exists(src_dir):
            log(f"  Not found: {src_dir} — skipping")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy2(src, dst)
            log(f"  ✅ {fname}")
            total += 1

    log(f"\nSynced {total} files to local project.")


def sync_from_zip(zip_path):
    log(f"Extracting zip: {zip_path}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RESULTS_DIR)
    files = os.listdir(RESULTS_DIR)
    log(f"Extracted {len(files)} files to {RESULTS_DIR}")
    for f in sorted(files):
        log(f"  ✅ {f}")


def auto_find_drive():
    for path in DRIVE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def print_summary():
    log("\n── Local results summary ──")
    if not os.path.exists(RESULTS_DIR):
        log("  No results yet.")
        return
    import json
    rows = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, f)) as fp:
                try:
                    d = json.load(fp)
                    rows.append(d)
                except Exception:
                    pass
    if not rows:
        log("  No JSON result files found.")
        return
    print(f"\n  {'Model':<14} {'Noise':<10} {'Val F1':<10} {'Test F1'}")
    print(f"  {'-'*46}")
    for r in sorted(rows, key=lambda x: (x.get('model',''), x.get('noise_level',''))):
        print(f"  {r.get('model','?'):<14} {r.get('noise_level','?'):<10} "
              f"{r.get('best_val_f1', r.get('val_f1_macro', '—')):<10} "
              f"{r.get('test_f1_macro','—')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive_path", type=str, default=None,
                        help="Path to hindiMix-noisy folder in Google Drive")
    parser.add_argument("--zip", type=str, default=None,
                        help="Path to results_tables.zip downloaded from Colab")
    args = parser.parse_args()

    if args.zip:
        sync_from_zip(args.zip)
    elif args.drive_path:
        sync_from_folder(args.drive_path)
    else:
        # Auto-detect Drive
        drive_path = auto_find_drive()
        if drive_path:
            log(f"Auto-detected Drive at: {drive_path}")
            sync_from_folder(drive_path)
        else:
            log("Google Drive folder not found automatically.")
            log("Options:")
            log("  1. python scripts/sync_from_drive.py --drive_path \"C:/Users/abhij/Google Drive/hindiMix-noisy\"")
            log("  2. python scripts/sync_from_drive.py --zip path/to/results_tables.zip")
            log("\nLooked in:")
            for p in DRIVE_CANDIDATES:
                log(f"  {p}")
            sys.exit(1)

    print_summary()
