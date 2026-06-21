"""
Axis 4 runner: energy-based swing-up with each model's learned energy.

For every (env, energy-capable model, seed) it trains the model and runs the
energy-shaping swing-up on the true plant, recording how high the tip gets and
the success rate. A true-energy "Oracle" is run per environment as the upper
bound. Black-box models (MLP, GP) have no energy function and are reported as
N/A in the aggregation step.

Usage:
    python scripts/run_control.py --models lnn hnn lgp --envs acrobot pendubot \
        --seeds 0 1 2 --budget 4000 --epochs 120
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

from src.envs import build_env
from src.experiment import load_dataset, split_dataset, train_model
from src.control import evaluate_control

GAINS = dict(ke=1.0, kp=0.3, kd=1.5, u_max=10.0, horizon=4000)
ACTUATION = {"acrobot": [0.0, 1.0], "pendubot": [1.0, 0.0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["lnn", "hnn", "lgp"])
    p.add_argument("--envs", nargs="+", default=["acrobot", "pendubot"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--budget", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--n_trials", type=int, default=5)
    args = p.parse_args()

    runs_dir = REPO_ROOT / "outputs" / "control" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with initialize(version_base=None, config_path="../configs"):
        datasets = {e: load_dataset(REPO_ROOT, e.capitalize()) for e in args.envs}

        # Oracle (true energy) upper bound, once per environment.
        for env_name in args.envs:
            tag = f"Oracle_on_{env_name.capitalize()}"
            out = runs_dir / f"{tag}.json"
            if out.exists():
                continue
            cfg = compose(config_name="train", overrides=[f"env={env_name}"])
            env = build_env(cfg)
            B = torch.tensor(ACTUATION[env_name])
            res = evaluate_control(env, env.get_true_energy, B,
                                   n_trials=args.n_trials, **GAINS)
            res.update({"model": "Oracle", "env": cfg.env.name, "seed": -1})
            out.write_text(json.dumps(res, indent=2))
            print(f"{tag}: height={res['mean_max_height']:.3f} success={res['success_rate']:.0%}")

        # Learned models.
        grid = [(m, e, s) for e in args.envs for m in args.models for s in args.seeds]
        for i, (model_name, env_name, seed) in enumerate(grid, 1):
            cfg = compose(config_name="train", overrides=[
                f"model={model_name}", f"env={env_name}",
                f"seed={seed}", f"epochs={args.epochs}",
            ])
            tag = f"{cfg.model.name}_on_{cfg.env.name}_s{seed}"
            out = runs_dir / f"{tag}.json"
            if out.exists():
                print(f"[{i}/{len(grid)}] {tag} -- done, skip")
                continue

            t0 = time.time()
            train_data, _, n_used = split_dataset(
                datasets[env_name], cfg.data.test_split, args.budget, seed)
            model = train_model(cfg, train_data, verbose=False)

            env = build_env(cfg)
            B = torch.tensor(ACTUATION[env_name])
            res = evaluate_control(env, model.get_energy, B,
                                   n_trials=args.n_trials, seed=seed, **GAINS)
            res.update({"model": cfg.model.name, "env": cfg.env.name,
                        "seed": seed, "n_train": n_used})
            out.write_text(json.dumps(res, indent=2))
            print(f"[{i}/{len(grid)}] {tag}: height={res['mean_max_height']:.3f} "
                  f"success={res['success_rate']:.0%} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDone. Control records in {runs_dir}")


if __name__ == "__main__":
    main()
