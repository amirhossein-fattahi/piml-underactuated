"""
Runs the full benchmark grid: every (model x env x seed x data-budget)
combination is trained and evaluated, and one result record per run is written
to outputs/results/runs/. This is the experiment driver for the paper.

Examples:
    # quick demo grid (Acrobot, 2 models, 2 seeds, 3 budgets)
    python scripts/run_experiments.py --envs acrobot --seeds 0 1 \
        --budgets 250 1000 4000 --epochs 150

    # full grid for the paper (both robots, more seeds, full data too)
    python scripts/run_experiments.py --envs acrobot pendubot --seeds 0 1 2 3 4 \
        --budgets 250 1000 4000 full --epochs 300
"""
import sys
import json
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from hydra import initialize, compose

from src.experiment import (
    generate_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)


def parse_budget(b):
    """'full' / 'all' / 'none' -> None (use all data); otherwise an int."""
    return None if str(b).lower() in ("full", "all", "none", "null") else int(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["mlp", "lnn"])
    p.add_argument("--envs", nargs="+", default=["acrobot"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--budgets", nargs="+", default=["250", "1000", "4000"])
    p.add_argument("--epochs", type=int, default=150)
    args = p.parse_args()

    budgets = [parse_budget(b) for b in args.budgets]
    runs_dir = REPO_ROOT / "outputs" / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO_ROOT / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    grid = [(m, e, s, b) for e in args.envs for m in args.models
            for s in args.seeds for b in budgets]
    print(f"Total runs: {len(grid)}\n")

    with initialize(version_base=None, config_path="../configs"):
        # Ensure each environment has a dataset (generate once if missing).
        for env in args.envs:
            path = data_dir / f"{env}_dataset.pt"
            if not path.exists():
                cfg = compose(config_name="train", overrides=[f"env={env}"])
                print(f"Generating dataset for {env}...")
                torch.save(generate_dataset(cfg), path)

        datasets = {e: torch.load(data_dir / f"{e}_dataset.pt") for e in args.envs}

        for i, (model, env, seed, budget) in enumerate(grid, 1):
            b_over = "null" if budget is None else str(budget)
            cfg = compose(
                config_name="train",
                overrides=[
                    f"model={model}", f"env={env}", f"seed={seed}",
                    f"epochs={args.epochs}", f"data.n_train={b_over}",
                ],
            )
            train_data, test_data, n_used = split_dataset(
                datasets[env], cfg.data.test_split, budget, seed
            )

            tag = f"{cfg.model.name}_on_{cfg.env.name}_s{seed}_n{n_used}"
            out_file = runs_dir / f"{tag}.json"
            if out_file.exists():
                print(f"[{i}/{len(grid)}] {tag} -- already done, skipping", flush=True)
                continue

            t0 = time.time()
            print(f"[{i}/{len(grid)}] {tag} ...", flush=True)

            m = train_model(cfg, train_data, verbose=False)
            results, _ = evaluate_model(cfg, m, test_data, n_used)
            out_file.write_text(json.dumps(results, indent=2))

            print(f"    one-step={results['one_step_mse']:.2e} "
                  f"valid_steps={results['valid_steps']:.0f} "
                  f"drift={results['energy_max_drift']:.2e} "
                  f"diverge={results['divergence_rate']:.0%} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\nDone. {len(grid)} run records in {runs_dir}")
    print("Next: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
