"""
Evaluates a trained dynamics model on three axes:
  Axis 1: one-step prediction accuracy (held-out test split)
  Axis 2: long-horizon rollout stability (autoregressive free-swing rollout)
  Axis 3: energy drift over the rollout (true energy should be conserved at zero torque)
(Axis 4, downstream swing-up control, is added in a later phase.)

Usage:
    python scripts/evaluate.py model=mlp env=acrobot
"""
import sys
import math
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hydra
from omegaconf import DictConfig
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.envs import build_env
from src.models import build_model
from src.utils.metrics import calculate_energy_drift


def angle_aware_sq_error(a, b):
    """Per-step mean squared error with angle wrapping on the first two dims."""
    diff = a - b
    ang = (diff[..., :2] + math.pi) % (2 * math.pi) - math.pi
    vel = diff[..., 2:]
    return torch.mean(torch.cat([ang, vel], dim=-1) ** 2, dim=-1)


def wrap_state_angles(state):
    state = state.clone()
    state[0] = (state[0] + math.pi) % (2 * math.pi) - math.pi
    state[1] = (state[1] + math.pi) % (2 * math.pi) - math.pi
    return state


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def evaluate(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    print(f"Evaluating {cfg.model.name} on {cfg.env.name}...")

    env = build_env(cfg)
    model = build_model(cfg)
    weight_path = REPO_ROOT / "outputs" / "models" / f"{cfg.model.name}_on_{cfg.env.name}.pt"
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # ------------------------------------------------------------------
    # Axis 1: one-step prediction accuracy on the held-out test split
    # ------------------------------------------------------------------
    data_path = REPO_ROOT / "data" / "raw" / f"{cfg.env.name.lower()}_dataset.pt"
    dataset = torch.load(data_path)
    S, A, NS = dataset["states"], dataset["actions"], dataset["next_states"]
    n_total = S.shape[0]
    n_train = int(n_total * (1.0 - cfg.data.test_split))
    S_te, A_te, NS_te = S[n_train:], A[n_train:], NS[n_train:]

    with torch.no_grad():
        pred_te = model.predict_next_state(S_te, A_te)
    one_step_mse = angle_aware_sq_error(pred_te, NS_te).mean().item()

    # ------------------------------------------------------------------
    # Axes 2 & 3: autoregressive free-swing (zero-torque) rollouts.
    # True energy is conserved by the simulator, so any drift in the model's
    # predicted trajectory is a pure physics-violation signal.
    # ------------------------------------------------------------------
    horizon = cfg.eval.horizon
    n_rollouts = cfg.eval.n_rollouts
    zero_action = torch.zeros(1)

    rollout_errors, max_drifts, drift_vars = [], [], []
    first_true, first_pred = None, None  # kept for plotting

    for r in range(n_rollouts):
        s0 = env.reset()

        # True trajectory
        true_traj = [s0]
        st = s0
        for _ in range(horizon):
            st = env.step(st, zero_action)
            true_traj.append(st)
        true_traj = torch.stack(true_traj)

        # Model trajectory (autoregressive)
        pred_traj = [s0]
        st = s0
        with torch.no_grad():
            for _ in range(horizon):
                st = wrap_state_angles(model.predict_next_state(st, zero_action))
                pred_traj.append(st)
        pred_traj = torch.stack(pred_traj)

        per_step_err = angle_aware_sq_error(pred_traj, true_traj)
        rollout_errors.append(per_step_err.mean().item())

        energies = env.get_true_energy(pred_traj)
        drift = calculate_energy_drift(energies)
        max_drifts.append(drift["max_drift"])
        drift_vars.append(drift["variance"])

        if r == 0:
            first_true, first_pred = true_traj, pred_traj

    results = {
        "model": cfg.model.name,
        "env": cfg.env.name,
        "one_step_mse": one_step_mse,
        "rollout_mse": float(sum(rollout_errors) / len(rollout_errors)),
        "energy_max_drift": float(sum(max_drifts) / len(max_drifts)),
        "energy_variance": float(sum(drift_vars) / len(drift_vars)),
    }

    res_dir = REPO_ROOT / "outputs" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"{cfg.model.name}_on_{cfg.env.name}.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    # Diagnostic figure: q1 trajectory + energy over the first rollout
    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = range(first_true.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, first_true[:, 0].numpy(), label="true", lw=2)
    axes[0].plot(t, first_pred[:, 0].numpy(), "--", label="model", lw=2)
    axes[0].set_title(f"{cfg.model.name} on {cfg.env.name}: q1 rollout")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("q1 [rad]")
    axes[0].legend()
    axes[1].plot(t, env.get_true_energy(first_true).numpy(), label="true E", lw=2)
    axes[1].plot(t, env.get_true_energy(first_pred).numpy(), "--", label="model E", lw=2)
    axes[1].set_title("Total energy along rollout")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("energy [J]")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / f"{cfg.model.name}_on_{cfg.env.name}.png", dpi=120)
    plt.close(fig)

    print(f"Results for {cfg.model.name} on {cfg.env.name}:")
    print(f"  Axis 1  One-step MSE:    {results['one_step_mse']:.6e}")
    print(f"  Axis 2  Rollout MSE:     {results['rollout_mse']:.6e}")
    print(f"  Axis 3  Energy drift:    {results['energy_max_drift']:.6e} (var {results['energy_variance']:.3e})")
    print(f"Saved metrics to {res_path}")


if __name__ == "__main__":
    evaluate()
