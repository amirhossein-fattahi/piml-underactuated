# src/envs/pendubot.py
import torch
from .base_env import DoublePendulumBase

class PendubotEnv(DoublePendulumBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = "Pendubot"
        self.state_dim = 4
        self.action_dim = 1
        
        # The Pendubot is actuated ONLY at the first joint (shoulder)
        self.B = torch.tensor([1.0, 0.0], dtype=torch.float32)