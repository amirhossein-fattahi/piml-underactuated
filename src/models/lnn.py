import torch
import torch.nn as nn
from .base_model import BaseModel

class LagrangianNN(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Network to learn the Cholesky decomposition of the Mass Matrix M(q)
        # We learn L so that M = L * L^T, guaranteeing M is symmetric positive-definite!
        self.mass_net = nn.Sequential(
            nn.Linear(2, 64), # Input is just positions (q1, q2)
            nn.Softplus(),    # Softplus is smooth/differentiable
            nn.Linear(64, 3)  # Outputs 3 values for a 2x2 lower triangular matrix
        )
        
        # Network to learn Potential Energy V(q)
        self.potential_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Softplus(),
            nn.Linear(64, 1)  # Outputs a single scalar (Energy)
        )

    def _get_mass_matrix(self, q):
        L_params = self.mass_net(q)
        # Construct lower triangular matrix L
        L = torch.zeros((q.shape[0], 2, 2)) if q.dim() > 1 else torch.zeros((2, 2))
        L[..., 0, 0] = torch.exp(L_params[..., 0]) # Diagonal must be positive
        L[..., 1, 0] = L_params[..., 1]
        L[..., 1, 1] = torch.exp(L_params[..., 2])
        
        M = L @ L.transpose(-1, -2) # M = L * L^T
        return M

    def forward(self, state, action):
        """
        Uses the Euler-Lagrange equations to find accelerations.
        (This will require PyTorch torch.autograd to compute derivatives of M and V).
        """
        q = state[..., 0:2].requires_grad_(True)
        dq = state[..., 2:4]
        
        # Calculate learned matrices
        M = self._get_mass_matrix(q)
        V = self.potential_net(q)
        
        # NOTE: For your thesis, you will use torch.autograd here to take the 
        # derivative of M and V with respect to q and dq to form the Coriolis 
        # matrix and solve for ddq. 
        
        # [Placeholder for Euler-Lagrange autograd solver]
        ddq = torch.zeros_like(dq) 
        
        return torch.cat([dq, ddq], dim=-1)

    def get_energy(self, state):
        q = state[..., 0:2]
        dq = state[..., 2:4]
        
        M = self._get_mass_matrix(q)
        V = self.potential_net(q)
        
        # T = 0.5 * dq^T * M * dq
        T = 0.5 * torch.einsum('...i,...ij,...j->...', dq, M, dq)
        
        return T + V # Total energy!