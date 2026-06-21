"""
Lagrangian-informed Gaussian Process (the thesis-contribution model).

Design: a hybrid of physics structure and a data-driven correction.
  1. An internal Lagrangian Neural Network (LNN) provides a *physics mean*:
     accelerations that obey the Euler-Lagrange equations with a learned,
     positive-definite mass matrix, a learned potential, and the known
     actuation matrix B.
  2. A Gaussian Process then models the *residual* acceleration that the
     physics mean fails to capture (un-modelled effects, fitting error).

Prediction: ddq = ddq_LNN(q, dq, u) + GP_residual(features).

This keeps the strong physics prior of the LNN while letting the GP correct
what the structural model misses -- combining structure, flexibility, and
(via the GP) a notion of predictive uncertainty. It is trained from the same
data as every other model, so the comparison stays fair.
"""
import math

import torch
import torch.optim as optim

from .base_model import BaseModel
from .lnn import LagrangianNN
from .gp import _ExactGP1D, gp_features, GP_MAX_POINTS


def _angle_aware_mse(pred, target):
    diff = pred - target
    ang = (diff[..., :2] + math.pi) % (2 * math.pi) - math.pi
    return torch.mean(torch.cat([ang, diff[..., 2:]], dim=-1) ** 2)


class LagrangianGP(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.lnn = LagrangianNN(cfg)
        self.gp = [_ExactGP1D(), _ExactGP1D()]
        # Internal LNN training budget (kept modest; the GP cleans up the rest).
        self._lnn_epochs = cfg.get("lnn_epochs", 100)
        self._lr = cfg.get("learning_rate", 1e-3)
        self._batch_size = cfg.get("batch_size", 256)

    # ------------------------------------------------------------------
    def fit(self, train_data):
        S, A, NS = train_data

        # 1) Train the internal LNN physics mean (one-step prediction loss).
        self.lnn.train()
        opt = optim.Adam(self.lnn.parameters(), lr=self._lr)
        n = S.shape[0]
        bs = self._batch_size
        for _ in range(self._lnn_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                pred = self.lnn.predict_next_state(S[idx], A[idx])
                loss = _angle_aware_mse(pred, NS[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lnn.parameters(), 10.0)
                opt.step()
        self.lnn.eval()

        # 2) Fit the GP to the residual acceleration the LNN did not explain,
        #    on the same capped subsample of points.
        Sx, Ax, NSx = _resubsample(train_data)
        lnn_accel = self.lnn.forward(Sx, Ax)[..., 2:4].detach()
        true_accel = (NSx[..., 2:4] - Sx[..., 2:4]) / self.dt
        residual = true_accel - lnn_accel
        feats = gp_features(Sx, Ax)
        for d in range(2):
            self.gp[d].fit(feats, residual[..., d])

    # ------------------------------------------------------------------
    def forward(self, state, action):
        out = self.lnn.forward(state, action)          # [dq, ddq_LNN]
        dq, ddq_phys = out[..., 0:2], out[..., 2:4]

        feats = gp_features(state, action)
        flat = feats.reshape(-1, feats.shape[-1])
        resid = [self.gp[d].predict_mean(flat).reshape(feats.shape[:-1]) for d in range(2)]
        ddq = ddq_phys + torch.stack(resid, dim=-1).to(state.dtype)
        return torch.cat([dq, ddq], dim=-1)

    def get_energy(self, state):
        # Energy comes from the physics (LNN) component.
        return self.lnn.get_energy(state)


def _resubsample(train_data, seed=0):
    """Same subsampling rule as gp._subsample, but returns the raw
    (S, A, NS) rows so the LNN residual is computed on the GP's points."""
    S, A, NS = train_data
    n = S.shape[0]
    if n > GP_MAX_POINTS:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:GP_MAX_POINTS]
        S, A, NS = S[idx], A[idx], NS[idx]
    return S, A, NS
