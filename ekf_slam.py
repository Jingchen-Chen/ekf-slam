import numpy as np


class EKFSLAM:
    """
    Full-state EKF-SLAM for a 2-D differential-drive robot with point landmarks.

    State vector:  mu = [x, y, theta,  lx_0, ly_0,  lx_1, ly_1, ...]
    Dimensions:    3 + 2 * n_landmarks
    """

    def __init__(self, n_landmarks, Q_obs, R_motion):
        n = 3 + 2 * n_landmarks

        self.mu    = np.zeros(n)          # state mean
        self.Sigma = np.eye(n) * 1e6      # state covariance (landmarks start with huge uncertainty)
        self.Sigma[:3, :3] = np.zeros((3, 3))  # robot initial pose is known

        self.Q = Q_obs       # observation noise covariance  (2×2)
        self.R = R_motion    # motion noise covariance       (3×3)
        self.initialized  = [False] * n_landmarks
        self.n_landmarks  = n_landmarks

    # ── PREDICT ──────────────────────────────────────────────────────────────

    def predict(self, u, dt=0.1):
        """
        EKF prediction step.

        u = (v, w)  linear and angular velocity commands
        Applies the velocity motion model and propagates the covariance via
        the linearised Jacobian.
        """
        v, w = u
        x, y, th = self.mu[:3]
        n = len(self.mu)

        # --- 1. Apply motion model ---
        self.mu[0] += v * np.cos(th) * dt
        self.mu[1] += v * np.sin(th) * dt
        self.mu[2]  = self._wrap(self.mu[2] + w * dt)

        # --- 2. Motion Jacobian F (n×n), identity except robot block ---
        F = np.eye(n)
        F[0, 2] = -v * np.sin(th) * dt
        F[1, 2] =  v * np.cos(th) * dt

        # --- 3. Noise mapping matrix G (n×3) – noise only affects robot pose ---
        G = np.zeros((n, 3))
        G[:3, :3] = np.eye(3)

        # --- 4. Covariance propagation:  Σ = F Σ Fᵀ + G R Gᵀ ---
        self.Sigma = F @ self.Sigma @ F.T + G @ self.R @ G.T

    # ── UPDATE ───────────────────────────────────────────────────────────────

    def update(self, observations):
        """
        EKF update step.

        observations : list of (range, bearing, landmark_id) tuples
        First-time observations initialise the landmark; subsequent observations
        run the standard EKF correction.
        """
        for r_obs, b_obs, j in observations:
            idx = 3 + 2 * j  # index of landmark j in the state vector

            # --- Initialise landmark on first observation ---
            if not self.initialized[j]:
                self.mu[idx]     = self.mu[0] + r_obs * np.cos(b_obs + self.mu[2])
                self.mu[idx + 1] = self.mu[1] + r_obs * np.sin(b_obs + self.mu[2])
                self.initialized[j] = True
                continue  # skip correction on the initialisation step

            # --- 1. Predicted observation  h(μ) ---
            dx = self.mu[idx]     - self.mu[0]
            dy = self.mu[idx + 1] - self.mu[1]
            q  = dx ** 2 + dy ** 2
            sq = np.sqrt(q)

            z_hat = np.array([
                sq,
                self._wrap(np.arctan2(dy, dx) - self.mu[2])
            ])

            # --- 2. Observation Jacobian H  (2×n) ---
            n = len(self.mu)
            H = np.zeros((2, n))

            # Partial derivatives w.r.t. robot pose [x, y, θ]
            H[0, 0] = -dx / sq;  H[0, 1] = -dy / sq;  H[0, 2] =  0
            H[1, 0] =  dy / q;   H[1, 1] = -dx / q;   H[1, 2] = -1

            # Partial derivatives w.r.t. landmark position [lx, ly]
            H[0, idx]     =  dx / sq;  H[0, idx + 1] =  dy / sq
            H[1, idx]     = -dy / q;   H[1, idx + 1] =  dx / q

            # --- 3. Innovation covariance & Kalman gain ---
            S = H @ self.Sigma @ H.T + self.Q
            K = self.Sigma @ H.T @ np.linalg.inv(S)

            # --- 4. State and covariance update ---
            innovation    = np.array([r_obs, b_obs]) - z_hat
            innovation[1] = self._wrap(innovation[1])   # bearing wrap

            self.mu   += K @ innovation
            self.Sigma = (np.eye(n) - K @ H) @ self.Sigma
            self.mu[2] = self._wrap(self.mu[2])

    # ── UTILITY ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(a):
        """Wrap angle to (−π, π]."""
        return (a + np.pi) % (2 * np.pi) - np.pi