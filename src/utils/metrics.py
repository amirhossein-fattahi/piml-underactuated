# src/utils/metrics.py
import torch
import numpy as np

def calculate_rollout_error(true_trajectory, pred_trajectory):
    """
    Axis 2: Long-horizon rollout stability.
    Calculates the Mean Squared Error over an entire trajectory.
    """
    # Ensure inputs are tensors
    if not isinstance(true_trajectory, torch.Tensor):
        true_trajectory = torch.tensor(true_trajectory)
    if not isinstance(pred_trajectory, torch.Tensor):
        pred_trajectory = torch.tensor(pred_trajectory)
        
    mse_per_step = torch.mean((true_trajectory - pred_trajectory)**2, dim=-1)
    total_mse = torch.mean(mse_per_step)
    
    return total_mse.item()

def calculate_energy_drift(energies):
    """
    Axis 3: Energy drift over time.
    Calculates the variance and absolute drift of a system's energy.
    Physics-informed models should score near 0.0 here!
    """
    if not isinstance(energies, torch.Tensor):
        energies = torch.tensor(energies)
        
    energy_variance = torch.var(energies).item()
    max_drift = (torch.max(energies) - torch.min(energies)).item()
    
    return {
        "variance": energy_variance,
        "max_drift": max_drift
    }

def wrap_angle(theta):
    """
    Utility to keep angles cleanly bound between -pi and pi.
    Crucial for calculating accurate errors on rotating joints.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi