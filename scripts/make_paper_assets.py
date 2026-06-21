"""
Collects the final paper-ready artifacts into a tracked `paper_assets/` folder
(figures + CSV tables + a RESULTS.md summary), so they are version-controlled and
survive regeneration of the gitignored `outputs/` directory.

Usage (run after the benchmark + control experiments and their aggregators):
    python scripts/aggregate_results.py
    python scripts/aggregate_control.py
    python scripts/make_paper_assets.py
"""
import sys
import csv
import shutil
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"
ASSETS = REPO_ROOT / "paper_assets"


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _copy(patterns, dest):
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for pat in patterns:
        for f in sorted((OUT / "figures").glob(pat)):
            shutil.copy(f, dest / f.name)
            copied.append(f.name)
    return copied


def main():
    (ASSETS / "figures").mkdir(parents=True, exist_ok=True)

    figs = _copy(["sweep_*.png", "control_*.png"], ASSETS / "figures")

    # Copy raw CSV tables.
    for src in [OUT / "results" / "summary.csv", OUT / "control" / "control_summary.csv"]:
        if src.exists():
            shutil.copy(src, ASSETS / src.name)

    # Build RESULTS.md
    lines = ["# Benchmark results\n",
             "Physics-informed dynamics models for underactuated swing-up. "
             "Five models (Vanilla MLP, Lagrangian NN, Hamiltonian NN, Vanilla GP, "
             "Lagrangian GP) on Acrobot and Pendubot, across four axes.\n"]

    # --- Dynamics table at the largest budget per env ---
    dyn = _read_csv(OUT / "results" / "summary.csv")
    if dyn:
        envs = sorted({r["env"] for r in dyn})
        for env in envs:
            rows = [r for r in dyn if r["env"] == env]
            max_n = max(int(r["n_train"]) for r in rows)
            rows = [r for r in rows if int(r["n_train"]) == max_n]
            lines.append(f"\n## Dynamics — {env} (largest budget n_train={max_n})\n")
            lines.append("| Model | One-step MSE ↓ | Valid steps ↑ | Energy drift ↓ | Diverged |")
            lines.append("|---|---|---|---|---|")
            for r in sorted(rows, key=lambda x: x["model"]):
                lines.append(
                    f"| {r['model']} | {float(r['one_step_mse_mean']):.2e} | "
                    f"{float(r['valid_steps_mean']):.1f} | "
                    f"{float(r['energy_max_drift_mean']):.2e} | "
                    f"{float(r['divergence_rate_mean']):.0%} |")

    # --- Control table ---
    ctrl = _read_csv(OUT / "control" / "control_summary.csv")
    if ctrl:
        envs = sorted({r["env"] for r in ctrl})
        for env in envs:
            rows = [r for r in ctrl if r["env"] == env]
            lines.append(f"\n## Swing-up control (Axis 4) — {env}\n")
            lines.append("| Model | Mean tip height (0–1) | Success rate |")
            lines.append("|---|---|---|")
            order = (["Oracle"] + sorted(r["model"] for r in rows if r["model"] != "Oracle"))
            seen = set()
            for m in order:
                rs = [r for r in rows if r["model"] == m]
                if not rs or m in seen:
                    continue
                seen.add(m)
                lines.append(f"| {m} | {float(rs[0]['mean_max_height']):.3f} | "
                             f"{float(rs[0]['success_rate']):.0%} |")

    # --- Figures ---
    lines.append("\n## Figures\n")
    for f in figs:
        lines.append(f"![{f}](figures/{f})\n")

    (ASSETS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {ASSETS / 'RESULTS.md'}")
    print(f"Copied {len(figs)} figures and CSV tables into {ASSETS}")


if __name__ == "__main__":
    main()
