# tests/test_dynamics.py
import torch
import math
from src.envs.acrobot import AcrobotEnv

def test_acrobot_energy_conservation():
    """
    If we drop the pendulum with 0 torque, total energy (T + V) 
    should remain perfectly constant over time.
    """
    # Create a dummy config
    cfg = {"dt": 0.01, "gravity": 9.81, "m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "lc1": 0.5, "lc2": 0.5, "I1": 0.083, "I2": 0.083}
    env = AcrobotEnv(cfg)
    
    # Start dropped slightly to the side [q1, q2, dq1, dq2]
    state = torch.tensor([math.pi/4, 0.0, 0.0, 0.0], dtype=torch.float32)
    zero_action = torch.tensor([0.0], dtype=torch.float32)
    
    initial_energy = env.get_true_energy(state)

    # Let it swing for 100 steps under zero torque
    for _ in range(100):
        state = env.step(state, zero_action)

    final_energy = env.get_true_energy(state)

    assert state.shape == (4,), "State shape must remain consistent."
    # RK4 at dt=0.01 should conserve energy very tightly
    assert abs((initial_energy - final_energy).item()) < 1e-3, (
        f"Energy drifted by {(initial_energy - final_energy).item():.2e}; "
        "the simulator's gravity vector and energy function are inconsistent."
    )