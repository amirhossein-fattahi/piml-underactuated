"""
Trains a dynamics model to predict the next state from (state, action) via
one-step supervised learning. The loss is angle-aware (wraps angular errors to
[-pi, pi]) so the wrap-around at +/-pi does not corrupt the gradient.

Usage:
    python scripts/train.py model=mlp env=acrobot
    python scripts/train.py model=lnn env=acrobot
"""
import sys
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim

from src.models import build_model


def angle_aware_mse(pred, target):
    """MSE where the first two dims (angles) are compared modulo 2*pi."""
    diff = pred - target
    ang = (diff[..., :2] + math.pi) % (2 * math.pi) - math.pi
    vel = diff[..., 2:]
    return torch.mean(torch.cat([ang, vel], dim=-1) ** 2)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train_model(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    print(f"Training {cfg.model.name} on {cfg.env.name} data...")

    data_path = REPO_ROOT / "data" / "raw" / f"{cfg.env.name.lower()}_dataset.pt"
    dataset = torch.load(data_path)
    S, A, NS = dataset["states"], dataset["actions"], dataset["next_states"]

    # Hold out the tail as a test split for one-step accuracy (used by evaluate.py).
    n_total = S.shape[0]
    n_train = int(n_total * (1.0 - cfg.data.test_split))
    S, A, NS = S[:n_train], A[:n_train], NS[:n_train]

    model = build_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    bs = cfg.batch_size
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n_train)
        running = 0.0
        for i in range(0, n_train, bs):
            idx = perm[i : i + bs]
            s, a, ns = S[idx], A[idx], NS[idx]

            pred = model.predict_next_state(s, a)
            loss = angle_aware_mse(pred, ns)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)

        if epoch % 50 == 0 or epoch == cfg.epochs - 1:
            print(f"Epoch {epoch}/{cfg.epochs} | Loss: {running / n_train:.6f}")

    out_dir = REPO_ROOT / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_path = out_dir / f"{cfg.model.name}_on_{cfg.env.name}.pt"
    torch.save(model.state_dict(), weight_path)
    print(f"Training complete. Weights saved to {weight_path}")


if __name__ == "__main__":
    train_model()
