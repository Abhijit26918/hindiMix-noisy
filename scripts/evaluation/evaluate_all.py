"""
Unified evaluation script — compares all models across all noise levels.
Generates the main results table for the paper.

Run: python scripts/evaluation/evaluate_all.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results/tables"
FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all_results():
    """Load all JSON result files from results/tables/."""
    records = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
        records.append(data)
    return pd.DataFrame(records)


def make_results_table(df):
    """Pivot into model × noise_level table (paper Table 1)."""
    pivot = df.pivot_table(
        index="model",
        columns="noise_level",
        values="test_f1_macro",
        aggfunc="mean"
    )
    # Reorder columns
    col_order = [c for c in ["clean", "low", "medium", "high"] if c in pivot.columns]
    pivot = pivot[col_order]

    # Add degradation column (clean - high noise)
    if "clean" in pivot.columns and "high" in pivot.columns:
        pivot["degradation"] = pivot["clean"] - pivot["high"]

    pivot = pivot.round(4)
    print("\n=== MAIN RESULTS TABLE ===")
    print(pivot.to_string())
    pivot.to_csv(f"{RESULTS_DIR}/main_results_table.csv")
    print(f"\nSaved to {RESULTS_DIR}/main_results_table.csv")
    return pivot


def plot_robustness_curve(df):
    """Line plot: F1 vs noise level for each model (paper Figure 1)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    noise_order = ["clean", "low", "medium", "high"]
    noise_labels = {"clean": "Clean", "low": "Low Noise", "medium": "Medium Noise", "high": "High Noise"}

    models = df["model"].unique()
    colors = sns.color_palette("husl", len(models))

    for model, color in zip(models, colors):
        model_df = df[df["model"] == model]
        x_vals, y_vals = [], []
        for nl in noise_order:
            row = model_df[model_df["noise_level"] == nl]
            if not row.empty:
                x_vals.append(noise_labels.get(nl, nl))
                y_vals.append(row["test_f1_macro"].values[0])

        style = "-o" if model == "NoiseRobust-HateDetect" else "--s"
        lw = 2.5 if model == "NoiseRobust-HateDetect" else 1.5
        ax.plot(x_vals, y_vals, style, label=model, color=color, linewidth=lw)

    ax.set_xlabel("Noise Level", fontsize=13)
    ax.set_ylabel("F1 (macro)", fontsize=13)
    ax.set_title("Robustness to ASR Noise: All Models", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{FIGURES_DIR}/robustness_curve.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.show()


def main():
    if not os.listdir(RESULTS_DIR):
        print("[WARN] No result files found. Train models first.")
        print("  Run: python models/baselines/muril_trainer.py --noise_level clean")
        return

    df = load_all_results()
    print(f"Loaded {len(df)} result records from {len(df['model'].unique())} models")

    make_results_table(df)
    if len(df) > 2:
        plot_robustness_curve(df)


if __name__ == "__main__":
    main()
