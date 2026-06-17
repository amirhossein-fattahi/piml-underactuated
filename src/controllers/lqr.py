import numpy as np
import scipy.linalg

class LQRController:
    def __init__(self, A, B, Q, R):
        """
        A and B are the linearized dynamics matrices at the upright equilibrium.
        """
        self.K = self._compute_lqr(A, B, Q, R)
        self.target_state = np.array([0.0, 0.0, 0.0, 0.0]) # Upright position

    def _compute_lqr(self, A, B, Q, R):
        # Solve the continuous-time Algebraic Riccati Equation
        P = scipy.linalg.solve_continuous_are(A, B, Q, R)
        # Compute the optimal gain matrix K
        K = np.linalg.inv(R) @ B.T @ P
        return K

    def get_action(self, state):
        # u = -K * (x - x_target)
        error = state - self.target_state
        
        # Wrap angles to handle multiple rotations
        error[0] = (error[0] + np.pi) % (2 * np.pi) - np.pi
        error[1] = (error[1] + np.pi) % (2 * np.pi) - np.pi
        
        action = -self.K @ error
        return action