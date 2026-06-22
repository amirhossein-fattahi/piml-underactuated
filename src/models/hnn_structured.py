import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class StructuredHNN(BaseModel):
    """
    Structured Hamiltonian Neural Network with a *learned* configuration-
    dependent mass matrix -- the fair counterpart to the naive HNN.

    The naive HNN assumes p := q_dot (an identity mass matrix), which is badly
    violated by a double pendulum and makes it fail. Here we instead learn a
    positive-definite mass matrix M(q) and a potential V(q), define the
    canonical momentum p = M(q) q_dot, and use the Hamiltonian

        H(q, p) = 1/2 p^T M(q)^{-1} p + V(q),

    following Hamilton's equations
        q_dot = dH/dp = M^{-1} p,
        p_dot = -dH/dq + B u.
    The acceleration follows from p_dot = M q_ddot + M_dot q_dot:
        q_ddot = M^{-1} ( -dH/dq|_p + B u - M_dot q_dot ).

    This gives the Hamiltonian model the same representational power as the LNN.
    Keeping both the naive and structured variants turns the naive HNN's failure
    into a controlled ablation: the failure is due to the identity-mass
    assumption, not to Hamiltonian methods per se.
    """

    EPS = 1e-2

    def __init__(self, cfg):
        super().__init__(cfg)
        actuation = cfg.get("actuation", [0.0, 1.0])
        self.register_buffer("B", torch.as_tensor(actuation, dtype=torch.float32))

        hidden = 64
        self.mass_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 3),
        )
        self.potential_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def _mass(self, q):
        p = self.mass_net(q)
        l1 = F.softplus(p[..., 0])
        l2 = F.softplus(p[..., 2])
        lo = p[..., 1]
        zero = torch.zeros_like(l1)
        L = torch.stack([torch.stack([l1, zero], dim=-1),
                         torch.stack([lo, l2], dim=-1)], dim=-2)
        eye = torch.eye(2, dtype=L.dtype, device=L.device)
        return L @ L.transpose(-1, -2) + self.EPS * eye

    def forward(self, state, action):
        # Hamilton's equations with the canonical momentum p = M(q) q_dot reduce,
        # via the Legendre transform, to the same equations of motion as the
        # Lagrangian formulation. We evaluate them in the numerically stable form
        # M q_ddot = B u - g(q) + 1/2 d/dq(q_dot^T M q_dot) - M_dot q_dot,
        # using -dH/dq|_p = 1/2 q_dot^T (dM/dq) q_dot - dV/dq.
        dq = state[..., 2:4]
        with torch.enable_grad():
            q = state[..., 0:2].detach().requires_grad_(True)
            create_graph = self.training

            M = self._mass(q)
            V = self.potential_net(q)
            g = torch.autograd.grad(
                V.sum(), q, create_graph=create_graph, retain_graph=True)[0]

            def dentry(a, b):
                return torch.autograd.grad(
                    M[..., a, b].sum(), q,
                    create_graph=create_graph, retain_graph=True)[0]
            g00, g01, g11 = dentry(0, 0), dentry(0, 1), dentry(1, 1)
            row0 = torch.stack([g00, g01], dim=-1)
            row1 = torch.stack([g01, g11], dim=-1)
            dMdq = torch.stack([row0, row1], dim=-2)   # (..., k, a, b)

        Mdot = torch.einsum("...kab,...k->...ab", dMdq, dq)
        quad = torch.einsum("...a,...iab,...b->...i", dq, dMdq, dq)
        tau = action * self.B
        rhs = (tau - g + 0.5 * quad
               - torch.einsum("...ib,...b->...i", Mdot, dq)).unsqueeze(-1)
        ddq = torch.linalg.solve(M, rhs).squeeze(-1)
        return torch.cat([dq, ddq], dim=-1)

    def get_energy(self, state):
        q = state[..., 0:2]
        dq = state[..., 2:4]
        M = self._mass(q)
        V = self.potential_net(q).squeeze(-1)
        T = 0.5 * torch.einsum("...i,...ij,...j->...", dq, M, dq)
        return T + V
