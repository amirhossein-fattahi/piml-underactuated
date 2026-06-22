"""
Polynomial (ridge) regression dynamics model -- a classic, cheap system-
identification baseline. It regresses joint accelerations on a degree-2
polynomial expansion of physically meaningful features
[sin q1, cos q1, sin q2, cos q2, q_dot1, q_dot2, u] via closed-form ridge
least squares. No iterative training, no physics structure.
"""
import torch

from .base_model import BaseModel
from .gp import gp_features


def _poly2(z):
    """Degree-2 polynomial features (with bias) of a base feature vector z."""
    n = z.shape[-1]
    terms = [torch.ones_like(z[..., :1]), z]
    for i in range(n):
        for j in range(i, n):
            terms.append((z[..., i] * z[..., j]).unsqueeze(-1))
    return torch.cat(terms, dim=-1)


class PolynomialDynamics(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ridge = cfg.get("ridge", 1e-4)
        self.W = None  # (n_features, 2)

    def fit(self, train_data):
        S, A, NS = train_data
        X = _poly2(gp_features(S, A)).double()
        Y = ((NS[..., 2:4] - S[..., 2:4]) / self.dt).double()
        f = X.shape[-1]
        A_mat = X.t() @ X + self.ridge * torch.eye(f, dtype=torch.float64)
        self.W = torch.linalg.solve(A_mat, X.t() @ Y)  # (f, 2)

    def forward(self, state, action):
        X = _poly2(gp_features(state, action)).double()
        ddq = (X @ self.W).to(state.dtype)
        dq = state[..., 2:4]
        return torch.cat([dq, ddq], dim=-1)

    def get_energy(self, state):
        return torch.full(state.shape[:-1], float("nan"))
