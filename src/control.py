"""
Axis 4: downstream control via energy-based swing-up.

An energy-shaping controller pumps energy into the *true* plant using an energy
function provided by a model:

    u = k_e (E_top - E_model(x)) * dq_actuated  -  k_p q_actuated  -  k_d dq_actuated   (clipped)

E_top is the energy of the upright equilibrium, evaluated with the *same* energy
function (so any constant offset cancels). The controller therefore only needs a
model's energy estimate -- physics-informed models (LNN, HNN, Lagrangian-GP)
provide one natively, while black-box models (MLP, GP) cannot, which is itself a
result. We measure how close the tip gets to fully upright.
"""
import math

import torch

UPRIGHT = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float32)


def tip_height_norm(env, state):
    """Normalised tip height in [0, 1]: 0 = hanging down, 1 = fully upright."""
    q1, q2 = state[..., 0], state[..., 1]
    y = -env.l1 * torch.cos(q1) - env.l2 * torch.cos(q1 + q2)
    return ((y + (env.l1 + env.l2)) / (2 * (env.l1 + env.l2)))


def run_swingup(env, energy_fn, B, horizon=1000, ke=0.5, kp=0.5, kd=1.0,
                u_max=8.0, init_state=None):
    """Runs one energy-shaping swing-up on the true plant. Returns a dict with
    the best (max) normalised tip height reached and a success flag."""
    with torch.no_grad():
        e_top = torch.as_tensor(energy_fn(UPRIGHT)).reshape(())
        state = UPRIGHT * 0 if init_state is None else init_state.clone()
        # start hanging down with a tiny nudge so dq_actuated != 0
        if init_state is None:
            state = torch.tensor([0.05, 0.0, 0.0, 0.0])

        max_h = tip_height_norm(env, state).item()
        for _ in range(horizon):
            e = torch.as_tensor(energy_fn(state)).reshape(())
            q_act = (B * state[0:2]).sum()
            dq_act = (B * state[2:4]).sum()

            u = ke * (e_top - e) * dq_act - kp * q_act - kd * dq_act
            u = torch.clamp(u, -u_max, u_max).reshape(1)

            state = env.step(state, u)
            max_h = max(max_h, tip_height_norm(env, state).item())

    # Success = the tip reached near-vertical (continuous max_height is the
    # primary, less knife-edge, metric).
    return {"max_height": float(max_h), "success": bool(max_h > 0.9)}


# Fixed, deterministic start states (shared across all models for fairness).
# They perturb the shoulder angle q1 while keeping the elbow q2 ~ 0, the regime
# where the energy-shaping controller is reliable so that failures reflect the
# model's energy, not the controller.
FIXED_INITS = [
    [0.05, 0.0, 0.0, 0.0],
    [0.10, 0.0, 0.0, 0.0],
    [0.15, 0.0, 0.0, 0.0],
    [-0.10, 0.0, 0.0, 0.0],
]


def evaluate_control(env, energy_fn, B, n_trials=None, horizon=1000, seed=0, **gains):
    """Averages swing-up over the fixed set of start states."""
    heights, successes = [], []
    for s in FIXED_INITS:
        r = run_swingup(env, energy_fn, B, horizon=horizon,
                        init_state=torch.tensor(s), **gains)
        heights.append(r["max_height"])
        successes.append(r["success"])
    return {
        "mean_max_height": float(sum(heights) / len(heights)),
        "success_rate": float(sum(successes) / len(successes)),
    }
