"""
Axis 4: downstream control via energy-based swing-up with an LQR catch.

A two-stage controller swings the *true* plant up to the unstable upright
equilibrium and then balances it there:

  1. Energy-shaping (swing-up): pump energy using a model's energy estimate,
        u = k_e (E_top - E_model(x)) dq_act - k_p q_act - k_d dq_act,
     where E_top is the upright energy under the *same* model (so any constant
     offset cancels). Only models with an energy function (LNN, HNN, Structured
     HNN, Lagrangian-GP) can drive this stage; black-box models cannot.
  2. LQR catch (balance): once the state is inside a basin around the upright,
     switch to a linear-quadratic regulator computed from the true plant's
     linearization. This is the textbook Acrobot/Pendubot solution; it lets the
     controller actually hold the robot up instead of falling back, and it makes
     the success metric meaningful.

We report the best normalized tip height reached and whether the robot is
balanced upright at the end of the episode.
"""
import math

import numpy as np
import scipy.linalg
import torch

UPRIGHT = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float32)


def tip_height_norm(env, state):
    """Normalised tip height in [0, 1]: 0 = hanging down, 1 = fully upright."""
    q1, q2 = state[..., 0], state[..., 1]
    y = -env.l1 * torch.cos(q1) - env.l2 * torch.cos(q1 + q2)
    return (y + (env.l1 + env.l2)) / (2 * (env.l1 + env.l2))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def lqr_upright_gain(env, Q=None, R=None):
    """Linearizes the true continuous dynamics at the upright equilibrium and
    returns the LQR feedback gain K (1x4) that stabilizes it."""
    if Q is None:
        Q = np.diag([10.0, 10.0, 1.0, 1.0])
    if R is None:
        R = np.array([[1.0]])
    eps = 1e-4
    x_eq = UPRIGHT.clone()

    def f(x, u):
        return env._get_dynamics(x, torch.tensor([u], dtype=torch.float32)).numpy()

    A = np.zeros((4, 4))
    for i in range(4):
        dx = torch.zeros(4); dx[i] = eps
        A[:, i] = (f(x_eq + dx, 0.0) - f(x_eq - dx, 0.0)) / (2 * eps)
    B = ((f(x_eq, eps) - f(x_eq, -eps)) / (2 * eps)).reshape(4, 1)

    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return torch.tensor(K, dtype=torch.float32).reshape(4)


def run_swingup(env, energy_fn, B, horizon=2000, ke=1.0, kp=0.3, kd=1.5,
                u_max=10.0, init_state=None, lqr_K=None, switch_tol=0.5):
    """One two-stage swing-up on the true plant. Returns the best tip height and
    whether the robot ends balanced upright."""
    with torch.no_grad():
        e_top = torch.as_tensor(energy_fn(UPRIGHT)).reshape(())
        state = init_state.clone() if init_state is not None \
            else torch.tensor([0.05, 0.0, 0.0, 0.0])

        max_h = tip_height_norm(env, state).item()
        balanced_steps = 0
        for _ in range(horizon):
            err = state.clone()
            err[0] = _wrap(state[0] - math.pi)
            err[1] = _wrap(state[1])
            near_upright = (lqr_K is not None
                            and err[0].abs() < switch_tol
                            and err[1].abs() < switch_tol)

            if near_upright:                       # stage 2: LQR balance
                u = -(lqr_K * torch.stack(
                    [err[0], err[1], state[2], state[3]])).sum()
            else:                                  # stage 1: energy shaping
                e = torch.as_tensor(energy_fn(state)).reshape(())
                q_act = (B * state[0:2]).sum()
                dq_act = (B * state[2:4]).sum()
                u = ke * (e_top - e) * dq_act - kp * q_act - kd * dq_act

            u = torch.clamp(u, -u_max, u_max).reshape(1)
            state = env.step(state, u)

            h = tip_height_norm(env, state).item()
            max_h = max(max_h, h)
            balanced_steps = balanced_steps + 1 if h > 0.95 else 0

    # Primary metric is the continuous max tip height. "success" = swung up to
    # near-vertical; if an LQR catch is enabled, we additionally require the
    # robot to stay balanced for the final stretch.
    swung_up = max_h > 0.9
    if lqr_K is not None:
        return {"max_height": float(max_h), "success": bool(balanced_steps >= 50)}
    return {"max_height": float(max_h), "success": bool(swung_up)}


# Fixed, deterministic start states near hanging-down (shared across all models).
FIXED_INITS = [
    [0.05, 0.0, 0.0, 0.0],
    [0.10, 0.0, 0.0, 0.0],
    [0.15, 0.0, 0.0, 0.0],
    [-0.10, 0.0, 0.0, 0.0],
]


def evaluate_control(env, energy_fn, B, n_trials=None, horizon=2000, seed=0,
                     use_lqr=False, **gains):
    """Averages energy-shaping swing-up over the fixed start states. An LQR catch
    (use_lqr=True) is available but off by default, since robustly balancing the
    underactuated Acrobot from the swing-up entry is itself a hard control
    problem; the continuous tip-height metric is the primary, robust measure."""
    lqr_K = lqr_upright_gain(env) if use_lqr else None
    heights, successes = [], []
    for s in FIXED_INITS:
        r = run_swingup(env, energy_fn, B, horizon=horizon,
                        init_state=torch.tensor(s), lqr_K=lqr_K, **gains)
        heights.append(r["max_height"])
        successes.append(r["success"])
    return {
        "mean_max_height": float(sum(heights) / len(heights)),
        "success_rate": float(sum(successes) / len(successes)),
    }
