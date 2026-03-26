import numpy as np


class Simulation:
    def __init__(self):
        # Ground-truth landmark positions
        self.landmarks = np.array([
            [5, 5], [10, 2], [8, 8],
            [2, 8], [12, 6], [4, 12]
        ], dtype=float)

        # Noise covariance matrices
        self.R_motion = np.diag([0.1, 0.1, np.deg2rad(2)]) ** 2   # motion noise
        self.Q_obs   = np.diag([0.2, np.deg2rad(3)]) ** 2          # observation noise

        self.max_range = 7.0  # sensor maximum range (metres)

    def move_robot(self, state_true, u, dt=0.1):
        """Propagate true robot state with additive Gaussian motion noise."""
        v, w = u
        x, y, th = state_true
        noise = np.random.multivariate_normal([0, 0, 0], self.R_motion)

        x  += (v + noise[0]) * np.cos(th) * dt
        y  += (v + noise[1]) * np.sin(th) * dt
        th += (w + noise[2]) * dt
        return np.array([x, y, th])

    def observe(self, state_true):
        """
        Return noisy range-bearing observations for all landmarks within max_range.
        Each observation is a tuple (range, bearing, landmark_id).
        """
        observations = []
        x, y, th = state_true
        for i, (lx, ly) in enumerate(self.landmarks):
            dx, dy = lx - x, ly - y
            r = np.hypot(dx, dy)
            if r < self.max_range:
                bearing  = np.arctan2(dy, dx) - th
                r_noisy  = r       + np.random.normal(0, np.sqrt(self.Q_obs[0, 0]))
                b_noisy  = bearing + np.random.normal(0, np.sqrt(self.Q_obs[1, 1]))
                observations.append((r_noisy, b_noisy, i))
        return observations