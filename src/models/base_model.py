import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.state_dim = cfg.get("state_dim", 4)
        self.action_dim = cfg.get("action_dim", 1)
        self.dt = cfg.get("dt", 0.01)

    def forward(self, state, action):
        """
        Must return the predicted accelerations (ddq).
        """
        raise NotImplementedError

    def predict_next_state(self, state, action):
        """
        Uses RK4 integration (just like your true environment) to predict 
        the next state based on the learned accelerations.
        """
        # Runge-Kutta 4 integration
        k1 = self.forward(state, action)
        k2 = self.forward(state + 0.5 * self.dt * k1, action)
        k3 = self.forward(state + 0.5 * self.dt * k2, action)
        k4 = self.forward(state + self.dt * k3, action)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state

    def get_energy(self, state):
        """
        Optional: Returns Total Energy = Kinetic + Potential.
        Only physics-informed models (like LNN) can compute this natively!
        """
        raise NotImplementedError