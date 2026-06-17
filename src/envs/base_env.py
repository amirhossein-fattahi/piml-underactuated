# src/envs/base_env.py
import torch
import math

class DoublePendulumBase:
    def __init__(self, cfg):
        """
        Initializes the double pendulum physics engine using the YAML config.
        """
        self.dt = cfg.get("dt", 0.01)
        
        # Physical parameters (defaulting to standard values if not in config)
        self.m1 = cfg.get("m1", 1.0)
        self.m2 = cfg.get("m2", 1.0)
        self.l1 = cfg.get("l1", 1.0)
        self.l2 = cfg.get("l2", 1.0)
        self.lc1 = cfg.get("lc1", 0.5)  # Center of mass
        self.lc2 = cfg.get("lc2", 0.5)
        self.I1 = cfg.get("I1", 0.083)  # Moment of inertia
        self.I2 = cfg.get("I2", 0.083)
        self.g = cfg.get("gravity", 9.81)
        
        # Actuation matrix (MUST be set by the child class: Acrobot or Pendubot)
        self.B = None 

    def _get_dynamics(self, state, action):
        """
        Computes accelerations q_ddot = M^-1 * (B*tau - C - G)
        """
        q1, q2, dq1, dq2 = state[0], state[1], state[2], state[3]
        tau = action[0]

        # 1. Mass Matrix M(q)
        d11 = self.m1 * self.lc1**2 + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * torch.cos(q2)) + self.I1 + self.I2
        d12 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * torch.cos(q2)) + self.I2
        d21 = d12
        d22 = self.m2 * self.lc2**2 + self.I2
        
        # Determinant for analytic inversion of 2x2 matrix
        det = d11 * d22 - d12 * d21

        # 2. Coriolis/Centrifugal Matrix C(q, q_dot) * q_dot
        h = -self.m2 * self.l1 * self.lc2 * torch.sin(q2)
        c1 = h * dq2**2 + 2 * h * dq1 * dq2
        c2 = -h * dq1**2

        # 3. Gravity Vector G(q) = dV/dq.  Note cos(x - pi/2) == sin(x).
        # phi1 must include BOTH links' contribution (it is dV/dq1), so the
        # second-link term sin(q1+q2) appears in phi1 as well as phi2.
        phi2 = self.m2 * self.lc2 * self.g * torch.sin(q1 + q2)
        phi1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * torch.sin(q1) + phi2

        # 4. Input Torque mapping
        # self.B determines if the torque goes to joint 1 (Pendubot) or joint 2 (Acrobot)
        u1 = self.B[0] * tau
        u2 = self.B[1] * tau

        # Compute accelerations (M^-1 * (tau - C - G))
        term1 = u1 - c1 - phi1
        term2 = u2 - c2 - phi2

        ddq1 = (d22 * term1 - d12 * term2) / det
        ddq2 = (-d21 * term1 + d11 * term2) / det

        return torch.tensor([dq1, dq2, ddq1, ddq2], dtype=torch.float32)

    def step(self, state, action):
        """
        Advances the simulation by one timestep using RK4 integration for high accuracy.
        """
        # Runge-Kutta 4th order method
        k1 = self._get_dynamics(state, action)
        k2 = self._get_dynamics(state + 0.5 * self.dt * k1, action)
        k3 = self._get_dynamics(state + 0.5 * self.dt * k2, action)
        k4 = self._get_dynamics(state + self.dt * k3, action)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Wrap angles between -pi and pi to keep data clean
        next_state[0] = torch.remainder(next_state[0] + math.pi, 2*math.pi) - math.pi
        next_state[1] = torch.remainder(next_state[1] + math.pi, 2*math.pi) - math.pi
        
        return next_state

    def sample_random_action(self):
        """Returns a random torque between -2.0 and 2.0 Nm for data generation."""
        return torch.rand(1) * 4.0 - 2.0

    def reset(self, state=None):
        """
        Returns an initial state. If `state` is None, samples broadly across the
        state space (angles in [-pi, pi], modest velocities) for good data coverage.
        Convention: q=0 is hanging straight down; q1=pi, q2=0 is fully upright.
        """
        if state is not None:
            return torch.as_tensor(state, dtype=torch.float32).clone()
        q = (torch.rand(2) * 2.0 - 1.0) * math.pi
        dq = (torch.rand(2) * 2.0 - 1.0) * 1.0
        return torch.cat([q, dq]).to(torch.float32)

    def get_true_energy(self, state):
        """
        Ground-truth total mechanical energy E = T + V. Supports a single state
        of shape (4,) or a batch of shape (..., 4). The potential energy is
        referenced to the hanging-down configuration and is consistent (up to a
        constant) with the gravity vector used in _get_dynamics, so a zero-torque
        rollout conserves this quantity. This is the signal for Axis 3 (energy drift).
        """
        q1, q2 = state[..., 0], state[..., 1]
        dq1, dq2 = state[..., 2], state[..., 3]

        cos_q2 = torch.cos(q2)
        d11 = self.m1 * self.lc1**2 + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * cos_q2) + self.I1 + self.I2
        d12 = self.m2 * (self.lc2**2 + self.l1 * self.lc2 * cos_q2) + self.I2
        d22 = self.m2 * self.lc2**2 + self.I2

        # Kinetic energy T = 0.5 * dq^T M(q) dq
        T = 0.5 * (d11 * dq1**2 + 2 * d12 * dq1 * dq2 + d22 * dq2**2)
        # Potential energy V (zero at hanging-down q=0)
        V = -(self.m1 * self.lc1 + self.m2 * self.l1) * self.g * torch.cos(q1) \
            - self.m2 * self.lc2 * self.g * torch.cos(q1 + q2)
        return T + V