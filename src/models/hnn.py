import torch
import torch.nn as nn
from .base_model import BaseModel


class HamiltonianNN(BaseModel):
    """
    Hamiltonian Neural Network (Greydanus et al. 2019), canonical form.

    Learns a single scalar Hamiltonian H(q, p) and follows the symplectic flow

        dq/dt =  dH/dp
        dp/dt = -dH/dq + B u

    Energy is conserved by construction at zero input: along the flow
    dH/dt = dH/dq * dq/dt + dH/dp * dp/dt = 0.

    To stay on the same (q, q_dot) interface as the other models, this uses the
    common "identity-mass" convention p := q_dot (i.e. it does NOT learn a
    configuration-dependent mass matrix the way the LNN does). That makes it a
    clean, weaker physics prior than the LNN -- a useful benchmark point on
    systems like the double pendulum whose true inertia depends on q.

    Physics priors embedded here: energy conservation via symplectic structure
    and the known actuation matrix B. Unlike the LNN it does not encode a
    positive-definite, q-dependent mass matrix.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        actuation = cfg.get("actuation", [0.0, 1.0])
        self.register_buffer("B", torch.as_tensor(actuation, dtype=torch.float32))

        hidden = 64
        self.h_net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        """Returns the symplectic time derivative [dq, dp] of the state."""
        with torch.enable_grad():
            z = state.detach().requires_grad_(True)        # (..., 4) = (q, p:=q_dot)
            H = self.h_net(z)
            dH = torch.autograd.grad(
                H.sum(), z, create_graph=self.training
            )[0]                                            # (..., 4)

        dHdq = dH[..., 0:2]
        dHdp = dH[..., 2:4]
        tau = action * self.B                              # actuation B u

        dq = dHdp
        dp = -dHdq + tau
        return torch.cat([dq, dp], dim=-1)

    def get_energy(self, state):
        """Total energy is the learned Hamiltonian H(q, p)."""
        return self.h_net(state).squeeze(-1)
