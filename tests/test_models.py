# tests/test_models.py
import torch
from src.models.mlp import VanillaMLP
from src.models.lnn import LagrangianNN

def test_model_forward_passes():
    """
    Ensures all models in the zoo accept (state, action) 
    and return predicted next states of the correct shape.
    """
    cfg = {"state_dim": 4, "action_dim": 1, "dt": 0.01}
    
    # Create models
    mlp = VanillaMLP(cfg)
    lnn = LagrangianNN(cfg)
    
    # Create a dummy batch of 32 states and actions
    batch_size = 32
    dummy_states = torch.randn(batch_size, 4)
    dummy_actions = torch.randn(batch_size, 1)
    
    # Test Black-Box
    mlp_out = mlp.predict_next_state(dummy_states, dummy_actions)
    assert mlp_out.shape == (batch_size, 4), "MLP output shape is incorrect."
    
    # Test Physics-Informed
    lnn_out = lnn.predict_next_state(dummy_states, dummy_actions)
    assert lnn_out.shape == (batch_size, 4), "LNN output shape is incorrect."

def test_lnn_energy_output():
    """
    Ensures the LNN can output scalar energy values.
    """
    cfg = {"state_dim": 4, "action_dim": 1, "dt": 0.01}
    lnn = LagrangianNN(cfg)
    
    dummy_state = torch.randn(1, 4)
    energy = lnn.get_energy(dummy_state)
    
    assert energy.shape == (1, 1) or energy.dim() == 0, "Energy must be a scalar value."