"""
Shared experiment logic (training + the three dynamics-evaluation axes), used
by both the single-run scripts and the grid runner so that every result is
computed with *identical* code. Keeping this in one place is what makes the
benchmark numbers comparable across models.
"""
import math
import random

import numpy as np
import torch
import torch.optim as optim

from .envs import build_env
from .models import build_model

# Fixed seed for the evaluation rollouts so that *every* model is tested on the
# exact same set of free-swing initial conditions (fair comparison).
EVAL_SEED = 12345


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------- losses/utils
def _angle_aware(diff):
    """Wrap the first two (angular) components of a state difference to [-pi, pi]."""
    ang = (diff[..., :2] + math.pi) % (2 * math.pi) - math.pi
    vel = diff[..., 2:]
    return torch.cat([ang, vel], dim=-1)


def angle_aware_mse(pred, target):
    return torch.mean(_angle_aware(pred - target) ** 2)


def angle_aware_sq_error(pred, target):
    return torch.mean(_angle_aware(pred - target) ** 2, dim=-1)


def wrap_state_angles(state):
    state = state.clone()
    state[..., 0] = (state[..., 0] + math.pi) % (2 * math.pi) - math.pi
    state[..., 1] = (state[..., 1] + math.pi) % (2 * math.pi) - math.pi
    return state


# ---------------------------------------------------------------- data handling
def generate_dataset(cfg):
    """Rolls out the true simulator under random torques to build a
    (state, action, next_state) dataset. Returns a dict of tensors."""
    set_seed(cfg.seed)
    env = build_env(cfg)
    states, actions, next_states = [], [], []
    for _ in range(cfg.data.num_episodes):
        state = env.reset()
        for _ in range(cfg.data.episode_length):
            action = env.sample_random_action()
            nxt = env.step(state, action)
            states.append(state.clone())
            actions.append(action.clone())
            next_states.append(nxt.clone())
            state = nxt
    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "next_states": torch.stack(next_states),
    }


def load_dataset(repo_root, env_name):
    path = repo_root / "data" / "raw" / f"{env_name.lower()}_dataset.pt"
    return torch.load(path)


def split_dataset(dataset, test_split, n_train=None, seed=0):
    """
    Splits into a fixed held-out test set (the tail) and a training pool (the
    head). If n_train is given, randomly subsamples that many points from the
    pool (seeded), so the data-budget sweep uses well-covered subsets rather
    than just the first few episodes. Returns (train, test, n_train_used).
    """
    S, A, NS = dataset["states"], dataset["actions"], dataset["next_states"]
    n = S.shape[0]
    n_pool = int(n * (1.0 - test_split))

    pool = (S[:n_pool], A[:n_pool], NS[:n_pool])
    test = (S[n_pool:], A[n_pool:], NS[n_pool:])

    if n_train is not None and n_train < n_pool:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n_pool, generator=g)[:n_train]
        pool = (pool[0][idx], pool[1][idx], pool[2][idx])

    return pool, test, pool[0].shape[0]


# ---------------------------------------------------------------- train/eval
def train_model(cfg, train_data, verbose=True):
    set_seed(cfg.seed)
    S, A, NS = train_data
    n = S.shape[0]
    bs = cfg.batch_size

    model = build_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            pred = model.predict_next_state(S[idx], A[idx])
            loss = angle_aware_mse(pred, NS[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running += loss.item() * len(idx)

        if verbose and (epoch % 50 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch}/{cfg.epochs} | loss {running / n:.6f}")

    model.eval()
    return model


# Caps keep diverged rollouts from producing inf/1e9 values that would dominate
# seed-averages and break log-scale plots. A double pendulum is chaotic, so raw
# long-horizon state-MSE saturates; we therefore lead with Valid Prediction Time.
STATE_DIVERGE = 1e2          # |state| beyond this (or non-finite) = blown up
ROLLOUT_ERR_CAP = 1e3        # cap on per-rollout state MSE
ENERGY_CAP = 1e3             # cap on per-rollout energy drift


def _rollout_model(model, s0, horizon, zero_action):
    """Autoregressive free-swing rollout with a divergence guard. Returns the
    trajectory collected so far and whether it blew up (NaN/inf or huge state,
    e.g. a singular learned mass matrix)."""
    traj = [s0]
    st = s0
    diverged = False
    for _ in range(horizon):
        try:
            nxt = model.predict_next_state(st, zero_action).detach()
        except Exception:
            diverged = True
            break
        if not torch.isfinite(nxt).all() or nxt.abs().max() > STATE_DIVERGE:
            diverged = True
            break
        st = wrap_state_angles(nxt)
        traj.append(st)
    return torch.stack(traj), diverged


def evaluate_model(cfg, model, test_data, n_train_used):
    """Computes the dynamics axes and returns (results_dict, plot_data).

    Axis 1: one-step MSE (lower better)
    Axis 2: valid prediction time in steps -- how long the rollout tracks the
            truth before per-step error exceeds a threshold (higher better);
            robust to chaos, unlike raw long-horizon MSE.
    Axis 3: energy drift over the rollout (lower better).
    Plus a divergence rate (fraction of rollouts that blew up).
    """
    S_te, A_te, NS_te = test_data

    pred_te = model.predict_next_state(S_te, A_te).detach()
    one_step_mse = angle_aware_sq_error(pred_te, NS_te).mean().item()

    env = build_env(cfg)
    set_seed(EVAL_SEED)
    horizon = cfg.eval.horizon
    vpt_thresh = cfg.eval.get("vpt_threshold", 1.0)
    zero_action = torch.zeros(1)

    rollout_errors, max_drifts, valid_steps, n_diverged = [], [], [], 0
    first_true, first_pred = None, None

    for r in range(cfg.eval.n_rollouts):
        s0 = env.reset()

        true_traj = [s0]
        st = s0
        for _ in range(horizon):
            st = env.step(st, zero_action)
            true_traj.append(st)
        true_traj = torch.stack(true_traj)

        pred_traj, diverged = _rollout_model(model, s0, horizon, zero_action)
        L = pred_traj.shape[0]
        per_step = angle_aware_sq_error(pred_traj, true_traj[:L])

        # Valid prediction time: first step where tracking error exceeds the
        # threshold (index 0 is the shared start, so search from step 1).
        exceed = (per_step[1:] > vpt_thresh).nonzero()
        vpt = (exceed[0].item() + 1) if len(exceed) > 0 else (L - 1)

        if diverged:
            n_diverged += 1
            rollout_errors.append(ROLLOUT_ERR_CAP)
            max_drifts.append(ENERGY_CAP)
        else:
            rollout_errors.append(min(per_step.mean().item(), ROLLOUT_ERR_CAP))
            energies = env.get_true_energy(pred_traj)
            max_drifts.append(min((energies.max() - energies.min()).item(), ENERGY_CAP))
        valid_steps.append(vpt)

        if r == 0:
            first_true, first_pred = true_traj, pred_traj

    results = {
        "model": cfg.model.name,
        "env": cfg.env.name,
        "seed": int(cfg.seed),
        "n_train": int(n_train_used),
        "one_step_mse": one_step_mse,
        "valid_steps": float(np.mean(valid_steps)),
        "energy_max_drift": float(np.mean(max_drifts)),
        "rollout_mse": float(np.mean(rollout_errors)),
        "divergence_rate": n_diverged / cfg.eval.n_rollouts,
    }
    return results, (first_true, first_pred)
