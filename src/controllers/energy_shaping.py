import numpy as np

class EnergyShapingController:
    def __init__(self, target_energy, k_energy=1.0):
        self.target_energy = target_energy # Energy of the system when perfectly upright
        self.k_energy = k_energy # Gain for energy injection

    def get_action(self, state, current_energy, dq_actuated):
        """
        Pumps energy into the system based on the energy error.
        dq_actuated is the velocity of the joint with the motor (Joint 1 or 2).
        """
        energy_error = current_energy - self.target_energy
        
        # Inject torque proportional to the energy error and the velocity 
        # of the actuated joint (Collocated partial feedback linearization)
        action = -self.k_energy * energy_error * dq_actuated
        
        # Clip action to motor limits
        return np.clip(action, -5.0, 5.0)