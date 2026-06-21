"""
Gaussian-process dynamics models (exact GP, implemented from scratch in torch
so the project needs no extra dependency).

The GP regresses the joint accelerations ddq from a feature vector built from
(state, action). Angles enter as (sin, cos) so the kernel respects their
periodicity. Acceleration targets are obtained from the data by finite
differences, ddq ~= (v_next - v) / dt.

GP models are fit by maximizing the marginal likelihood (a different training
path from the gradient-trained NNs), so they expose a `.fit(train_data)` method
that src.experiment.train_model calls instead of the Adam loop. Evaluation is
identical to every other model, keeping the benchmark fair.
"""
import math

import torch

from .base_model import BaseModel

# Exact GPs are O(n^3); cap the training set so fitting stays fast and stable.
GP_MAX_POINTS = 1000


def gp_features(state, action):
    """(state, action) -> 7-D features [sin q1, cos q1, sin q2, cos q2, dq1, dq2, tau]."""
    q1, q2 = state[..., 0], state[..., 1]
    dq1, dq2 = state[..., 2], state[..., 3]
    tau = action[..., 0]
    return torch.stack(
        [torch.sin(q1), torch.cos(q1), torch.sin(q2), torch.cos(q2), dq1, dq2, tau],
        dim=-1,
    )


class _ExactGP1D:
    """Single-output exact GP with an ARD-RBF kernel, fit by marginal likelihood.
    Computations are in float64 for numerical stability of the Cholesky."""

    def __init__(self, jitter=1e-4):
        self.jitter = jitter
        self.X = None
        self.alpha = None
        self.log_ls = None
        self.log_out = None

    @staticmethod
    def _kernel(A, B, log_ls, log_out):
        ls = torch.exp(log_ls)
        out = torch.exp(log_out)
        Aw, Bw = A / ls, B / ls
        d2 = (Aw ** 2).sum(-1, keepdim=True) - 2 * Aw @ Bw.t() + (Bw ** 2).sum(-1).unsqueeze(0)
        return out * torch.exp(-0.5 * d2.clamp_min(0.0))

    def fit(self, X, y, steps=25, lr=0.1):
        X, y = X.double(), y.double()
        n, f = X.shape
        log_ls = torch.zeros(f, dtype=torch.float64, requires_grad=True)
        log_out = torch.zeros((), dtype=torch.float64, requires_grad=True)
        log_noise = torch.tensor(math.log(0.1), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([log_ls, log_out, log_noise], lr=lr)
        eye = torch.eye(n, dtype=torch.float64)

        for _ in range(steps):
            opt.zero_grad()
            K = self._kernel(X, X, log_ls, log_out) + (torch.exp(log_noise) + self.jitter) * eye
            L = torch.linalg.cholesky(K)
            a = torch.cholesky_solve(y.unsqueeze(-1), L)
            nlml = (0.5 * (y.unsqueeze(-1) * a).sum()
                    + torch.log(torch.diag(L)).sum()
                    + 0.5 * n * math.log(2 * math.pi))
            nlml.backward()
            opt.step()

        with torch.no_grad():
            K = self._kernel(X, X, log_ls, log_out) + (torch.exp(log_noise) + self.jitter) * eye
            L = torch.linalg.cholesky(K)
            self.alpha = torch.cholesky_solve(y.unsqueeze(-1), L)
            self.X = X
            self.log_ls = log_ls.detach()
            self.log_out = log_out.detach()

    def predict_mean(self, Xq):
        Kqx = self._kernel(Xq.double(), self.X, self.log_ls, self.log_out)
        return (Kqx @ self.alpha).squeeze(-1)


def _subsample(train_data, dt, seed=0):
    """Build GP features/accel targets, capping at GP_MAX_POINTS."""
    S, A, NS = train_data
    n = S.shape[0]
    if n > GP_MAX_POINTS:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)[:GP_MAX_POINTS]
        S, A, NS = S[idx], A[idx], NS[idx]
    X = gp_features(S, A)
    ddq = (NS[..., 2:4] - S[..., 2:4]) / dt
    return X, ddq


class VanillaGP(BaseModel):
    """Black-box GP dynamics model (no physics prior)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.gp = [_ExactGP1D(), _ExactGP1D()]

    def fit(self, train_data):
        X, ddq = _subsample(train_data, self.dt)
        for d in range(2):
            self.gp[d].fit(X, ddq[..., d])

    def forward(self, state, action):
        feats = gp_features(state, action)
        flat = feats.reshape(-1, feats.shape[-1])
        preds = [self.gp[d].predict_mean(flat).reshape(feats.shape[:-1]) for d in range(2)]
        ddq = torch.stack(preds, dim=-1).to(state.dtype)
        dq = state[..., 2:4]
        return torch.cat([dq, ddq], dim=-1)

    def get_energy(self, state):
        # A black-box GP has no notion of energy.
        return torch.full(state.shape[:-1], float("nan"))
