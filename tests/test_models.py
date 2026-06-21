# tests/test_models.py
import torch
from src.models.mlp import VanillaMLP
from src.models.lnn import LagrangianNN
from src.models.hnn import HamiltonianNN
from src.models.gp import VanillaGP
from src.models.lgp import LagrangianGP


def test_gp_models_fit_and_predict():
    """The GP and Lagrangian-GP must fit via .fit() and then produce finite
    next-state predictions of the right shape."""
    torch.manual_seed(0)
    cfg = {"state_dim": 4, "action_dim": 1, "dt": 0.01, "actuation": [0.0, 1.0],
           "learning_rate": 1e-3, "batch_size": 64, "lnn_epochs": 10}
    n = 80
    S = torch.randn(n, 4)
    A = torch.randn(n, 1)
    NS = S + 0.01 * torch.randn(n, 4)
    train = (S, A, NS)

    for Model in (VanillaGP, LagrangianGP):
        model = Model(cfg)
        model.fit(train)
        out = model.predict_next_state(S[:5], A[:5]).detach()
        assert out.shape == (5, 4), f"{Model.__name__} output shape is wrong."
        assert torch.isfinite(out).all(), f"{Model.__name__} produced non-finite output."


def test_lnn_conserves_its_own_energy():
    """
    Structural guarantee of the Euler-Lagrange formulation: with zero input a
    conservative Lagrangian system conserves total energy. An *untrained* LNN
    must therefore keep its own get_energy() nearly constant along a free
    rollout -- this verifies the autograd Coriolis/gravity solver is correct,
    independent of any fitting.
    """
    torch.manual_seed(0)
    cfg = {"state_dim": 4, "action_dim": 1, "dt": 0.01, "actuation": [0.0, 1.0]}
    lnn = LagrangianNN(cfg)
    lnn.eval()

    state = torch.tensor([0.3, -0.2, 0.4, -0.3])
    zero_action = torch.zeros(1)
    e0 = lnn.get_energy(state).item()
    for _ in range(200):
        state = lnn.predict_next_state(state, zero_action).detach()
    e1 = lnn.get_energy(state).item()

    assert abs(e1 - e0) < 1e-4, (
        f"Untrained LNN drifted energy by {abs(e1 - e0):.2e}; the Euler-Lagrange "
        "solver does not conserve energy and is likely implemented incorrectly."
    )

def test_hnn_conserves_its_own_energy():
    """
    Hamiltonian counterpart of the LNN test: the symplectic flow conserves H by
    construction, so an *untrained* HNN must keep its own Hamiltonian nearly
    constant along a zero-input rollout.
    """
    torch.manual_seed(0)
    cfg = {"state_dim": 4, "action_dim": 1, "dt": 0.01, "actuation": [0.0, 1.0]}
    hnn = HamiltonianNN(cfg)
    hnn.eval()

    state = torch.tensor([0.3, -0.2, 0.4, -0.3])
    zero_action = torch.zeros(1)
    e0 = hnn.get_energy(state).item()
    for _ in range(200):
        state = hnn.predict_next_state(state, zero_action).detach()
    e1 = hnn.get_energy(state).item()

    assert abs(e1 - e0) < 1e-4, (
        f"Untrained HNN drifted energy by {abs(e1 - e0):.2e}; the symplectic "
        "flow does not conserve the Hamiltonian and is likely implemented wrong."
    )


def test_model_forward_passes():
    """
    Ensures all models in the zoo accept (state, action)
    and return predicted next states of the correct shape.
    """
    cfg = {"state_dim": 4, "action_dim": 1, "dt": 0.01}

    # Create a dummy batch of 32 states and actions
    batch_size = 32
    dummy_states = torch.randn(batch_size, 4)
    dummy_actions = torch.randn(batch_size, 1)

    # Every model in the zoo must map (state, action) -> next state of shape (B, 4)
    for Model in (VanillaMLP, LagrangianNN, HamiltonianNN):
        model = Model(cfg)
        out = model.predict_next_state(dummy_states, dummy_actions)
        assert out.shape == (batch_size, 4), f"{Model.__name__} output shape is incorrect."

def test_lnn_energy_output():
    """
    Ensures the LNN can output scalar energy values.
    """
    cfg = {"state_dim": 4, "action_dim": 1, "dt": 0.01}
    lnn = LagrangianNN(cfg)
    
    dummy_state = torch.randn(1, 4)
    energy = lnn.get_energy(dummy_state)

    assert energy.numel() == 1, "Energy must be one scalar per state."

    # Batched energy: one scalar per state in the batch.
    batch_energy = lnn.get_energy(torch.randn(16, 4))
    assert batch_energy.shape == (16,), "Batched energy must have shape (batch,)."