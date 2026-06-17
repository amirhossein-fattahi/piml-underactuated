# src/envs/acrobot.py
import torch
from .base_env import DoublePendulumBase

class AcrobotEnv(DoublePendulumBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = "Acrobot"
        self.state_dim = 4
        self.action_dim = 1
        
        # The Acrobot is actuated ONLY at the second joint (elbow)
        self.B = torch.tensor([0.0, 1.0], dtype=torch.float32)