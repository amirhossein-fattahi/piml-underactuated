"""
Generates a dynamics dataset of (state, action, next_state) transitions by
rolling out the true physics simulator under random torques.

Usage:
    python scripts/generate_data.py env=acrobot
    python scripts/generate_data.py env=pendubot
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra
from omegaconf import DictConfig
import torch

from src.envs import build_env


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def generate_data(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    print(f"Generating data for the {cfg.env.name} environment...")

    env = build_env(cfg)
    num_episodes = cfg.data.num_episodes
    ep_len = cfg.data.episode_length

    states, actions, next_states = [], [], []
    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(ep_len):
            action = env.sample_random_action()
            nxt = env.step(state, action)
            states.append(state.clone())
            actions.append(action.clone())
            next_states.append(nxt.clone())
            state = nxt

    states = torch.stack(states)
    actions = torch.stack(actions)
    next_states = torch.stack(next_states)

    out_dir = REPO_ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{cfg.env.name.lower()}_dataset.pt"
    torch.save(
        {"states": states, "actions": actions, "next_states": next_states},
        save_path,
    )

    print(f"Saved {states.shape[0]} transitions to {save_path}")


if __name__ == "__main__":
    generate_data()
