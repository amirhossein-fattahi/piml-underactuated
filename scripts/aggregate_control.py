"""
Aggregates the Axis-4 (swing-up control) records in outputs/control/runs/ into a
seed-averaged table + bar chart of swing-up success and tip height per model.

Usage:
    python scripts/aggregate_control.py
"""
import sys
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"Oracle": "#000000", "Lagrangian_NN": "#1f77b4",
          "Hamiltonian_NN": "#2ca02c", "Lagrangian_GP": "#d62728",
          "Vanilla_MLP": "#888888", "Vanilla_GP": "#9467bd"}


def main():
    runs_dir = REPO_ROOT / "outputs" / "control" / "runs"
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        print("No control records. Run scripts/run_control.py first.")
        return
    runs = [json.loads(f.read_text()) for f in files]

    group = defaultdict(list)
    for r in runs:
        group[(r["env"], r["model"])].append(r)

    envs = sorted({r["env"] for r in runs})
    res_dir = REPO_ROOT / "outputs" / "control"
    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / "control_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["env", "model", "n_seeds", "mean_max_height", "height_std",
                    "success_rate"])
        for (env, model), rs in sorted(group.items()):
            h = [r["mean_max_height"] for r in rs]
            s = [r["success_rate"] for r in rs]
            w.writerow([env, model, len(rs), np.mean(h), np.std(h), np.mean(s)])

    for env in envs:
        models = [m for (e, m) in group if e == env]
        # order: Oracle first, then the rest
        models = (["Oracle"] if "Oracle" in models else []) + sorted(m for m in models if m != "Oracle")

        print(f"\n### Swing-up control on {env} (seed-averaged)\n")
        print("| Model | Mean tip height (0-1) | Success rate |")
        print("|---|---|---|")
        heights, succ, labels = [], [], []
        for model in models:
            rs = group[(env, model)]
            h = np.mean([r["mean_max_height"] for r in rs])
            s = np.mean([r["success_rate"] for r in rs])
            print(f"| {model} | {h:.3f} +/- {np.std([r['mean_max_height'] for r in rs]):.3f} | {s:.0%} |")
            heights.append(h); succ.append(s); labels.append(model)

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, heights, color=[COLORS.get(m, "#555") for m in labels])
        ax.axhline(0.95, ls="--", color="grey", lw=1, label="swing-up threshold")
        ax.set_ylabel("mean max tip height (0=down, 1=up)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Energy-based swing-up on {env}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend()
        for b, s in zip(bars, succ):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{s:.0%}", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"control_{env}.png", dpi=120)
        plt.close(fig)
        print(f"\nSaved control_{env}.png")

    print(f"\nSaved {res_dir / 'control_summary.csv'}")


if __name__ == "__main__":
    main()
