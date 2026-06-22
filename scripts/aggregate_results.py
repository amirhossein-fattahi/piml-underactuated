"""
Aggregates all run records in outputs/results/runs/ into:
  - a seed-averaged comparison table (mean +/- std) at the largest data budget,
  - a data-budget sweep figure per axis (clean condition),
  - a clean-vs-noisy robustness table (real-world scenario),
  - a tidy CSV of every aggregated cell.

Usage:
    python scripts/aggregate_results.py
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

# (json key, column title, higher_is_better)
AXES = [
    ("one_step_mse", "Axis 1: One-step MSE (lower)", False),
    ("valid_steps", "Axis 2: Valid pred. steps (higher)", True),
    ("energy_max_drift", "Axis 3: Energy drift (lower)", False),
]
EXTRA = ("divergence_rate", "Diverged")
NOISE_BUDGET = 4000  # budget at which clean vs noisy is compared

COLORS = {"Vanilla_MLP": "#888888", "Polynomial": "#bcbd22",
          "Lagrangian_NN": "#1f77b4", "Hamiltonian_NN": "#e377c2",
          "Structured_HNN": "#2ca02c", "Vanilla_GP": "#9467bd",
          "Lagrangian_GP": "#d62728"}


def main():
    runs_dir = REPO_ROOT / "outputs" / "results" / "runs"
    files = sorted(runs_dir.glob("*.json"))
    if not files:
        print("No run records found. Run scripts/run_experiments.py first.")
        return
    runs = [json.loads(f.read_text()) for f in files]
    for r in runs:
        r.setdefault("noise", "clean")

    res_dir = REPO_ROOT / "outputs" / "results"
    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = [k for k, _, _ in AXES] + [EXTRA[0]]

    # group[(noise, env, model, n_train)] = list of per-seed run dicts
    group = defaultdict(list)
    for r in runs:
        group[(r["noise"], r["env"], r["model"], r["n_train"])].append(r)

    # ---- tidy CSV of every aggregated cell ----
    with open(res_dir / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["noise", "env", "model", "n_train", "n_seeds"]
                   + [f"{k}_mean" for k in metric_keys] + [f"{k}_std" for k in metric_keys])
        for (noise, env, model, n), rs in sorted(group.items()):
            means = [np.mean([r[k] for r in rs]) for k in metric_keys]
            stds = [np.std([r[k] for r in rs]) for k in metric_keys]
            w.writerow([noise, env, model, n, len(rs)] + means + stds)

    clean_envs = sorted({env for (noise, env, _, _) in group if noise == "clean"})

    # ---- clean comparison table + data-budget sweep (clean) ----
    for env in clean_envs:
        models = sorted({m for (no, e, m, _) in group if no == "clean" and e == env})
        budgets = sorted({n for (no, e, _, n) in group if no == "clean" and e == env})
        full = budgets[-1]

        print(f"\n### {env} (clean, largest budget n_train={full}, mean +/- std over seeds)\n")
        headers = ["Model"] + [name for _, name, _ in AXES] + [EXTRA[1]]
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join(["---"] * len(headers)) + "|")
        for model in models:
            rs = group.get(("clean", env, model, full), [])
            if not rs:
                continue
            cells = [model]
            for k, _, _ in AXES:
                vals = [r[k] for r in rs]
                cells.append(f"{np.mean(vals):.2e} +/- {np.std(vals):.0e}")
            cells.append(f"{np.mean([r[EXTRA[0]] for r in rs]):.0%}")
            print("| " + " | ".join(cells) + " |")

        if len(budgets) > 1:
            fig, axes = plt.subplots(1, len(AXES), figsize=(5 * len(AXES), 4))
            for ax, (k, name, higher) in zip(axes, AXES):
                for model in models:
                    xs, ys, es = [], [], []
                    for n in budgets:
                        rs = group.get(("clean", env, model, n), [])
                        if rs:
                            xs.append(n)
                            ys.append(np.mean([r[k] for r in rs]))
                            es.append(np.std([r[k] for r in rs]))
                    if xs:
                        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                                    label=model, color=COLORS.get(model))
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_xlabel("# training transitions")
                ax.set_title(name + (" ↑" if higher else " ↓"))
                ax.legend(fontsize=7)
            fig.suptitle(f"Data-budget sweep on {env} (clean)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"sweep_{env}.png", dpi=120)
            plt.close(fig)
            print(f"\nSaved sweep_{env}.png")

    # ---- clean vs noisy robustness table (at NOISE_BUDGET) ----
    noisy_exists = any(no == "noisy" for (no, _, _, _) in group)
    if noisy_exists:
        for env in clean_envs:
            models = sorted({m for (no, e, m, n) in group
                             if e == env and n == NOISE_BUDGET})
            if not models:
                continue
            print(f"\n### Noise robustness on {env} (n_train={NOISE_BUDGET}): "
                  f"one-step MSE / valid steps, clean -> noisy\n")
            print("| Model | One-step (clean) | One-step (noisy) | "
                  "Valid steps (clean) | Valid steps (noisy) |")
            print("|---|---|---|---|---|")
            for model in models:
                c = group.get(("clean", env, model, NOISE_BUDGET), [])
                nz = group.get(("noisy", env, model, NOISE_BUDGET), [])
                if not c or not nz:
                    continue

                def mean(rs, k):
                    return np.mean([r[k] for r in rs])
                print(f"| {model} | {mean(c,'one_step_mse'):.2e} | "
                      f"{mean(nz,'one_step_mse'):.2e} | "
                      f"{mean(c,'valid_steps'):.1f} | {mean(nz,'valid_steps'):.1f} |")

    print(f"\nSaved {res_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
