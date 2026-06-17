import torch
import torch.nn as nn
from .base_model import BaseModel

class VanillaMLP(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # A simple Multi-Layer Perceptron
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, int(self.state_dim / 2)) # Outputs ddq (2 values)
        )

    def forward(self, state, action):
        """
        Black-box prediction of accelerations.
        """
        x = torch.cat([state, action], dim=-1)
        ddq = self.net(x)
        
        # We must return a 4D vector [dq1, dq2, ddq1, ddq2] for the RK4 integrator
        dq = state[..., 2:4]
        return torch.cat([dq, ddq], dim=-1)
        
    def get_energy(self, state):
        # A black-box MLP does not know what energy is!
        return torch.tensor(float('nan'))