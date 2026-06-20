"""
Trains a single dynamics model on one environment (one seed, one data budget).
Thin wrapper around src.experiment so the metrics/logic match the grid runner.

Usage:
    python scripts/train.py model=mlp env=acrobot
    python scripts/train.py model=lnn env=acrobot seed=1 data.n_train=1000
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
from omegaconf import DictConfig

from src.experiment import load_dataset, split_dataset, train_model


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    print(f"Training {cfg.model.name} on {cfg.env.name} (seed={cfg.seed})...")

    dataset = load_dataset(REPO_ROOT, cfg.env.name)
    n_train = cfg.data.get("n_train", None)
    train_data, _, n_used = split_dataset(dataset, cfg.data.test_split, n_train, cfg.seed)
    print(f"Training on {n_used} transitions.")

    model = train_model(cfg, train_data)

    out_dir = REPO_ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_path = out_dir / f"{cfg.model.name}_on_{cfg.env.name}_s{cfg.seed}_n{n_used}.pt"
    torch.save(model.state_dict(), weight_path)
    print(f"Training complete. Weights saved to {weight_path}")


if __name__ == "__main__":
    main()
