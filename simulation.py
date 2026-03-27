"""
Ground-truth motion and noisy sensor simulation for 2-D SLAM.

Simulation class
----------------
- Generates a landmark layout centred near the robot's circular path.
- Propagates robot state with Gaussian motion noise.
- Produces noisy range-bearing observations.

Default trajectory
------------------
v = 1.0 m/s, omega = 0.25 rad/s  =>  radius R = v/omega = 4 m,
period T = 2*pi/omega = 25.1 s = 251 steps at dt=0.1.
Landmarks are placed at radii 2-7 m from the circle centre so the
robot always has 2-5 landmarks in its 8 m sensor range.
"""

import numpy as np


class Simulation:
    def __init__(self, *, max_range: float = 8.0,
                 motion_noise: tuple[float, float, float] = (0.05, 0.05, 1.0),
                 obs_noise: tuple[float, float] = (0.15, 2.0),
                 landmarks: np.ndarray | None = None):
        """
        Parameters
        ----------
        max_range    : sensor maximum range (metres)
        motion_noise : (sigma_v, sigma_v, sigma_w_deg) — motion noise std devs
        obs_noise    : (sigma_range, sigma_bearing_deg) — observation noise std devs
        landmarks    : (N, 2) array; uses the default layout if None
        """
        if landmarks is not None:
            self.landmarks = np.asarray(landmarks, dtype=float)
        else:
            # 8 landmarks arranged around a circle of radius R=4m centred at (4,0)
            # so the default circular trajectory (centre at ~(4,0)) always observes them
            cx, cy = 4.0, 0.0
            angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8 evenly spaced
            radii = [3.5, 5.5, 3.0, 6.0, 4.5, 3.8, 5.0, 4.2]
            self.landmarks = np.array(
                [[cx + r * np.cos(a), cy + r * np.sin(a)]
                 for r, a in zip(radii, angles)])

        sig_v1, sig_v2, sig_w_deg = motion_noise
        self.R_motion = np.diag([sig_v1**2, sig_v2**2,
                                 np.deg2rad(sig_w_deg)**2])

        sig_r, sig_b_deg = obs_noise
        self.Q_obs = np.diag([sig_r**2, np.deg2rad(sig_b_deg)**2])

        self.max_range = max_range

    # ── Motion model ─────────────────────────────────────────────────────────

    def move_robot(self, state_true: np.ndarray, u: tuple[float, float],
                   dt: float = 0.1) -> np.ndarray:
        """Propagate true robot state with additive Gaussian motion noise."""
        v, w = u
        x, y, th = state_true
        noise = np.random.multivariate_normal([0.0, 0.0, 0.0], self.R_motion)

        x += (v + noise[0]) * np.cos(th) * dt
        y += (v + noise[1]) * np.sin(th) * dt
        th = _wrap(th + (w + noise[2]) * dt)
        return np.array([x, y, th])

    # ── Observation model ────────────────────────────────────────────────────

    def observe(self, state_true: np.ndarray) -> list[tuple]:
        """
        Return noisy range-bearing observations for landmarks in sensor range.

        Each entry: (range, bearing, landmark_id)
        """
        observations = []
        x, y, th = state_true
        for i, (lx, ly) in enumerate(self.landmarks):
            dx, dy = lx - x, ly - y
            r = np.hypot(dx, dy)
            if r < self.max_range:
                bearing = _wrap(np.arctan2(dy, dx) - th)
                r_noisy = r + np.random.normal(0.0, np.sqrt(self.Q_obs[0, 0]))
                b_noisy = bearing + np.random.normal(0.0, np.sqrt(self.Q_obs[1, 1]))
                observations.append((r_noisy, b_noisy, i))
        return observations

    def observe_no_id(self, state_true: np.ndarray) -> list[tuple]:
        """
        Return noisy range-bearing observations WITHOUT landmark IDs.

        Each entry: (range, bearing)  — for unknown data association.
        """
        return [(r, b) for r, b, _ in self.observe(state_true)]


def _wrap(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
