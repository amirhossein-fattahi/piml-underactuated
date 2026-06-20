"""
Evaluates a single trained model (one seed, one data budget) on the three
dynamics axes and writes a result record + a diagnostic figure. Thin wrapper
around src.experiment.

Usage:
    python scripts/evaluate.py model=lnn env=acrobot seed=1 data.n_train=1000
"""
import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.envs import build_env
from src.models import build_model
from src.experiment import load_dataset, split_dataset, evaluate_model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print(f"Evaluating {cfg.model.name} on {cfg.env.name} (seed={cfg.seed})...")

    dataset = load_dataset(REPO_ROOT, cfg.env.name)
    n_train = cfg.data.get("n_train", None)
    _, test_data, n_used = split_dataset(dataset, cfg.data.test_split, n_train, cfg.seed)

    model = build_model(cfg)
    tag = f"{cfg.model.name}_on_{cfg.env.name}_s{cfg.seed}_n{n_used}"
    model.load_state_dict(torch.load(REPO_ROOT / "outputs" / "models" / f"{tag}.pt"))
    model.eval()

    results, (true_traj, pred_traj) = evaluate_model(cfg, model, test_data, n_used)

    runs_dir = REPO_ROOT / "outputs" / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{tag}.json").write_text(json.dumps(results, indent=2))

    # Diagnostic figure for this single run (pred_traj may be shorter if it
    # diverged, so plot each series over its own length).
    env = build_env(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(true_traj.shape[0]), true_traj[:, 0].numpy(), label="true", lw=2)
    axes[0].plot(range(pred_traj.shape[0]), pred_traj[:, 0].numpy(), "--", label="model", lw=2)
    axes[0].set_title(f"{tag}: q1 rollout")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("q1 [rad]"); axes[0].legend()
    axes[1].plot(range(true_traj.shape[0]), env.get_true_energy(true_traj).numpy(), label="true E", lw=2)
    axes[1].plot(range(pred_traj.shape[0]), env.get_true_energy(pred_traj).numpy(), "--", label="model E", lw=2)
    axes[1].set_title("Total energy along rollout")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("energy [J]"); axes[1].legend()
    fig.tight_layout()
    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{tag}.png", dpi=120)
    plt.close(fig)

    print(f"  Axis 1  One-step MSE: {results['one_step_mse']:.4e}")
    print(f"  Axis 2  Rollout MSE:  {results['rollout_mse']:.4e}")
    print(f"  Axis 3  Energy drift: {results['energy_max_drift']:.4e}")
    print(f"Saved result to {runs_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
