import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


class LagrangianNN(BaseModel):
    """
    Deep Lagrangian Network (DeLaN-style) for a 2-DoF underactuated system.

    Instead of regressing accelerations directly, it learns two physical
    functions and enforces the Euler-Lagrange equations of motion:

        M(q) ddq + c(q, dq) + g(q) = B u

    where
        M(q)      = L(q) L(q)^T + eps*I   (symmetric positive-definite by construction)
        g(q)      = dV/dq                 (V = learned potential energy)
        c(q, dq)  = dM/dt dq - 1/2 d/dq( dq^T M dq )   (Coriolis/centrifugal)
        B u       = actuation (B is the known, fixed actuation matrix)

    M and V are small MLPs; the Coriolis term and gravity are obtained from
    them via autograd, so the model is differentiable end-to-end and trains
    by backprop through these physics constraints.

    Physics priors embedded here: (i) symmetric PD mass matrix, (ii) a single
    scalar potential => conservative gravity, (iii) the known actuation
    structure B. A vanilla MLP has none of these.
    """

    EPS = 1e-2  # PD floor on the mass-matrix diagonal for stable inversion

    def __init__(self, cfg):
        super().__init__(cfg)

        # Known actuation matrix B (which joint the motor drives).
        # Acrobot -> [0, 1] (elbow), Pendubot -> [1, 0] (shoulder).
        actuation = cfg.get("actuation", [0.0, 1.0])
        self.register_buffer("B", torch.as_tensor(actuation, dtype=torch.float32))

        hidden = 64
        # Lower-triangular Cholesky factor of M(q): 3 params for a 2x2 matrix.
        self.mass_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 3),
        )
        # Potential energy V(q): scalar output.
        self.potential_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    # ------------------------------------------------------------------
    def _get_mass_matrix(self, q):
        """Builds M(q) = L L^T + eps*I (batched or single). No in-place ops,
        so it is safe for the double-backward used during training."""
        p = self.mass_net(q)                      # (..., 3)
        l_diag1 = F.softplus(p[..., 0])           # positive diagonal entries
        l_diag2 = F.softplus(p[..., 2])
        l_offd = p[..., 1]
        zero = torch.zeros_like(l_diag1)

        row0 = torch.stack([l_diag1, zero], dim=-1)
        row1 = torch.stack([l_offd, l_diag2], dim=-1)
        L = torch.stack([row0, row1], dim=-2)     # (..., 2, 2)

        M = L @ L.transpose(-1, -2)
        eye = torch.eye(2, dtype=M.dtype, device=M.device)
        return M + self.EPS * eye

    # ------------------------------------------------------------------
    def forward(self, state, action):
        """Returns the first-order time derivative [dq, ddq] of the state."""
        dq = state[..., 2:4]

        # q must be a differentiable leaf for the inner Jacobians, regardless
        # of whether we are inside a no_grad() block at inference time.
        with torch.enable_grad():
            q = state[..., 0:2].detach().requires_grad_(True)
            create_graph = self.training

            M = self._get_mass_matrix(q)          # (..., 2, 2)
            V = self.potential_net(q)             # (..., 1)

            # Gravity g(q) = dV/dq
            g = torch.autograd.grad(
                V.sum(), q, create_graph=create_graph, retain_graph=True
            )[0]                                   # (..., 2)

            # dM/dq: derivative of each unique entry of M w.r.t. q. Assembled
            # with stack (no in-place writes) so double-backward stays valid.
            def dentry(a, b):
                return torch.autograd.grad(
                    M[..., a, b].sum(), q,
                    create_graph=create_graph, retain_graph=True,
                )[0]                               # (..., 2) over k
            g00, g01, g11 = dentry(0, 0), dentry(0, 1), dentry(1, 1)

            # dMdq[..., k, a, b] = d M[..., a, b] / d q_k  (M symmetric => g01 reused)
            row_a0 = torch.stack([g00, g01], dim=-1)   # a=0: [dM00, dM01] over b
            row_a1 = torch.stack([g01, g11], dim=-1)   # a=1: [dM10, dM11] over b
            dMdq = torch.stack([row_a0, row_a1], dim=-2)  # (..., k, a, b)

        # Coriolis/centrifugal term c(q, dq):
        #   dMdt = sum_k (dM/dq_k) dq_k
        dMdt = torch.einsum("...kab,...k->...ab", dMdq, dq)
        #   quadratic term q_i = dq^T (dM/dq_i) dq
        quad = torch.einsum("...a,...iab,...b->...i", dq, dMdq, dq)
        c = torch.einsum("...ib,...b->...i", dMdt, dq) - 0.5 * quad

        # Actuation B u (B fixed/known, u is the scalar action)
        tau = action * self.B                      # (..., 2)

        # Solve M ddq = tau - c - g
        rhs = (tau - c - g).unsqueeze(-1)          # (..., 2, 1)
        ddq = torch.linalg.solve(M, rhs).squeeze(-1)

        return torch.cat([dq, ddq], dim=-1)

    # ------------------------------------------------------------------
    def get_energy(self, state):
        """Total mechanical energy E = T + V, computed natively from the
        learned M(q) and V(q). (A black-box MLP cannot do this.)"""
        q = state[..., 0:2]
        dq = state[..., 2:4]
        M = self._get_mass_matrix(q)
        V = self.potential_net(q).squeeze(-1)
        T = 0.5 * torch.einsum("...i,...ij,...j->...", dq, M, dq)
        return T + V
