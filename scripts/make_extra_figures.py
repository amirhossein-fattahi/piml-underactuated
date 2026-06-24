"""
Generates extra paper material from real runs:
  - per-model training-time table (computational cost),
  - an example free-swing rollout figure (true vs MLP vs LNN: angle + energy).

Usage:
    python scripts/make_extra_figures.py
"""
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra import initialize, compose

from src.envs import build_env
from src.experiment import (load_dataset, split_dataset, train_model,
                            set_seed, wrap_state_angles, EVAL_SEED)

MODELS = ["mlp", "poly", "gp", "lnn", "hnn", "shnn", "lgp"]


def main():
    with initialize(version_base=None, config_path="../configs"):
        ds = load_dataset(REPO_ROOT, "Acrobot")
        train_data, _, n_used = split_dataset(ds, 0.2, 1000, 0)

        timings, trained = {}, {}
        for m in MODELS:
            cfg = compose(config_name="train", overrides=[
                f"model={m}", "env=acrobot", "seed=0",
                "epochs=100", "data.n_train=1000"])
            t0 = time.time()
            model = train_model(cfg, train_data, verbose=False)
            timings[cfg.model.name] = time.time() - t0
            trained[cfg.model.name] = model

        print("\n=== Training time on Acrobot (n=1000, 100 epochs) ===")
        for name, t in sorted(timings.items(), key=lambda kv: kv[1]):
            print(f"{name:16s} {t:7.1f} s")

        # ---- example rollout figure: true vs MLP vs LNN ----
        cfg = compose(config_name="train", overrides=["env=acrobot"])
        env = build_env(cfg)
        set_seed(EVAL_SEED)
        s0 = env.reset()
        horizon, zero = 200, torch.zeros(1)

        def rollout(stepper):
            traj, st = [s0], s0
            for _ in range(horizon):
                st = wrap_state_angles(stepper(st))
                traj.append(st)
            return torch.stack(traj)

        true_traj = rollout(lambda st: env.step(st, zero))
        mlp_traj = rollout(lambda st: trained["Vanilla_MLP"].predict_next_state(st, zero).detach())
        lnn_traj = rollout(lambda st: trained["Lagrangian_NN"].predict_next_state(st, zero).detach())

        t = range(true_traj.shape[0])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(t, true_traj[:, 0], "k", lw=2, label="true")
        ax[0].plot(t, mlp_traj[:, 0], "--", color="#888", lw=2, label="MLP")
        ax[0].plot(t, lnn_traj[:, 0], "--", color="#1f77b4", lw=2, label="LNN")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("shoulder angle $q_1$ [rad]")
        ax[0].set_title("Free-swing trajectory"); ax[0].legend()

        ax[1].plot(t, env.get_true_energy(true_traj), "k", lw=2, label="true")
        ax[1].plot(t, env.get_true_energy(mlp_traj), "--", color="#888", lw=2, label="MLP")
        ax[1].plot(t, env.get_true_energy(lnn_traj), "--", color="#1f77b4", lw=2, label="LNN")
        ax[1].set_xlabel("step"); ax[1].set_ylabel("total energy [J]")
        ax[1].set_title("Energy along rollout"); ax[1].legend()
        fig.tight_layout()
        for d in [REPO_ROOT / "paper" / "figures", REPO_ROOT / "outputs" / "figures"]:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / "rollout_example.png", dpi=120)
        plt.close(fig)
        print("\nSaved rollout_example.png")


if __name__ == "__main__":
    main()
