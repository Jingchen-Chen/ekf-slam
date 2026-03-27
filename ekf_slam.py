"""
EKF-SLAM for a 2-D differential-drive robot with point landmarks.

Features
--------
- Dynamic state vector: landmarks added on-the-fly (no pre-allocation).
- Joseph-form covariance update for numerical stability.
- Maximum Likelihood data association (Mahalanobis distance) with a
  chi-squared gating threshold — no known landmark IDs required.
- Optional known-association mode for comparison / ground-truth evaluation.
- Loop closure detection: fires only when the robot revisits a landmark
  after traveling far away (meaningful covariance reduction).

State vector:  mu = [x, y, theta,  lx_0, ly_0,  lx_1, ly_1, ...]
"""

import numpy as np
from scipy.stats import chi2


class EKFSLAM:
    """Full-state EKF-SLAM with dynamic landmark management."""

    def __init__(self, Q_obs: np.ndarray, R_motion: np.ndarray,
                 known_association: bool = False,
                 gate_threshold: float | None = None,
                 min_obs_for_new_lm: int = 2):
        """
        Parameters
        ----------
        Q_obs             : 2x2 observation noise covariance (range, bearing)
        R_motion          : 3x3 motion noise covariance
        known_association : if True, observations carry ground-truth landmark IDs
        gate_threshold    : Mahalanobis distance^2 threshold for data association.
                            Default: chi2(df=2).ppf(0.99) ~= 9.21
        min_obs_for_new_lm: minimum confirmations required before a tentative
                            landmark is committed to the state vector (unknown
                            association only). Helps prevent ghost landmarks.
        """
        self.mu = np.zeros(3)
        self.Sigma = np.zeros((3, 3))

        self.Q = Q_obs.copy()
        self.R = R_motion.copy()

        self.n_landmarks = 0
        self.initialized: list[bool] = []
        # Maps internal landmark index -> state vector index
        self._lm_state_idx: list[int] = []
        # For known association: maps ground-truth ID -> internal index
        self._gt_to_internal: dict[int, int] = {}

        self.known_association = known_association
        self.gate_threshold = gate_threshold or chi2.ppf(0.99, df=2)
        self.min_obs_for_new_lm = min_obs_for_new_lm

        # Candidate buffer for unknown association: range/bearing sum for averaging
        # key = candidate_id, value = dict with 'obs', 'count', 'r_sum', 'b_sin', 'b_cos'
        self._candidates: dict[int, dict] = {}
        self._next_candidate_id = 0

        # Observation count per landmark
        self.observation_count: list[int] = []

        # Loop closure events
        self.loop_closures: list[dict] = []
        # Track the max robot pose covariance trace seen so far
        # (used to detect when a closure causes a significant reduction)
        self._max_robot_trace: float = 0.0

    # ── PREDICT ──────────────────────────────────────────────────────────────

    def predict(self, u: tuple[float, float], dt: float = 0.1):
        """
        EKF prediction step using velocity motion model.

        u = (v, w) — linear and angular velocity commands.
        """
        v, w = u
        th = self.mu[2]
        n = len(self.mu)

        # Motion model (mean update — noise-free)
        self.mu[0] += v * np.cos(th) * dt
        self.mu[1] += v * np.sin(th) * dt
        self.mu[2] = self._wrap(self.mu[2] + w * dt)

        # Jacobian F (n×n) — identity except robot pose block
        F = np.eye(n)
        F[0, 2] = -v * np.sin(th) * dt
        F[1, 2] = v * np.cos(th) * dt

        # Noise injection: only the robot pose rows receive process noise
        G = np.zeros((n, 3))
        G[:3, :3] = np.eye(3)

        self.Sigma = F @ self.Sigma @ F.T + G @ self.R @ G.T

        # Track the largest robot pose uncertainty seen (for loop closure scoring)
        robot_trace = float(np.trace(self.Sigma[:3, :3]))
        if robot_trace > self._max_robot_trace:
            self._max_robot_trace = robot_trace

    # ── UPDATE ───────────────────────────────────────────────────────────────

    def update(self, observations: list):
        """
        EKF update step.

        observations : list of tuples.
            Known association mode:   (range, bearing, landmark_id)
            Unknown association mode: (range, bearing)
        """
        for obs in observations:
            if self.known_association:
                r_obs, b_obs, gt_id = obs
                j = self._get_or_create_landmark(gt_id, r_obs, b_obs)
            else:
                r_obs, b_obs = obs[0], obs[1]
                j = self._associate(r_obs, b_obs)

            if j is None:
                continue  # observation is buffered as a candidate

            idx = self._lm_state_idx[j]

            # Skip EKF correction on the very first observation (just initialise)
            if not self.initialized[j]:
                self._init_landmark(j, r_obs, b_obs)
                continue

            self._ekf_correct(j, idx, r_obs, b_obs)

    # ── DATA ASSOCIATION ────────────────────────────────────────────────────

    def _associate(self, r_obs: float, b_obs: float) -> int | None:
        """
        Maximum Likelihood data association via Mahalanobis distance.

        For already-initialised landmarks, uses the EKF innovation gate.
        For new observations, accumulates them in a candidate buffer until
        min_obs_for_new_lm confirmations are collected, then commits the
        landmark to the state vector.

        Returns internal landmark index, or None if still in candidate buffer.
        """
        best_j = None
        best_dist = self.gate_threshold

        for j in range(self.n_landmarks):
            if not self.initialized[j]:
                continue
            idx = self._lm_state_idx[j]
            z_hat, H, S = self._compute_observation_model(idx)

            innovation = np.array([r_obs - z_hat[0],
                                   self._wrap(b_obs - z_hat[1])])
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            d2 = float(innovation @ S_inv @ innovation)

            if d2 < best_dist:
                best_dist = d2
                best_j = j

        if best_j is not None:
            self.observation_count[best_j] += 1
            return best_j

        # No match in the confirmed map: check candidates
        # Each candidate buffers observations and tries to match them
        best_cid = None
        best_cdist = self.gate_threshold

        for cid, cand in self._candidates.items():
            # Predicted candidate position (mean of buffered observations from
            # the robot pose at the time of first observation — approximate)
            cand_pos = np.array([cand['x'], cand['y']])
            dx = cand_pos[0] - self.mu[0]
            dy = cand_pos[1] - self.mu[1]
            q = max(dx**2 + dy**2, 1e-6)
            sq = np.sqrt(q)
            z_hat_c = np.array([sq, self._wrap(np.arctan2(dy, dx) - self.mu[2])])

            # Use a loose covariance based on position uncertainty of candidate
            S_c = np.diag([cand['sigma_r']**2 + self.Q[0, 0],
                           cand['sigma_b']**2 + self.Q[1, 1]])
            inno = np.array([r_obs - z_hat_c[0], self._wrap(b_obs - z_hat_c[1])])
            try:
                d2_c = float(inno @ np.linalg.inv(S_c) @ inno)
            except np.linalg.LinAlgError:
                continue
            if d2_c < best_cdist:
                best_cdist = d2_c
                best_cid = cid

        if best_cid is not None:
            # Update existing candidate with new observation (running mean of position)
            cand = self._candidates[best_cid]
            cand['count'] += 1
            # Update running mean of candidate position
            th = self.mu[2]
            x_obs = self.mu[0] + r_obs * np.cos(b_obs + th)
            y_obs = self.mu[1] + r_obs * np.sin(b_obs + th)
            alpha = 1.0 / cand['count']
            cand['x'] += alpha * (x_obs - cand['x'])
            cand['y'] += alpha * (y_obs - cand['y'])

            if cand['count'] >= self.min_obs_for_new_lm:
                # Promote candidate to confirmed landmark
                j = self._add_landmark_slot()
                # Use accumulated mean position
                idx = self._lm_state_idx[j]
                self.mu[idx] = cand['x']
                self.mu[idx + 1] = cand['y']
                self.initialized[j] = True
                self.observation_count[j] = cand['count']
                del self._candidates[best_cid]
                return j
            return None
        else:
            # Brand-new candidate
            th = self.mu[2]
            x_obs = self.mu[0] + r_obs * np.cos(b_obs + th)
            y_obs = self.mu[1] + r_obs * np.sin(b_obs + th)
            # Estimate positional uncertainty of candidate from obs noise
            sigma_r = np.sqrt(self.Q[0, 0])
            sigma_b = np.sqrt(self.Q[1, 1])
            # Propagate: sigma_pos ~ sqrt((sigma_r)^2 + (r*sigma_b)^2) + Sigma_robot
            r_approx = r_obs
            sigma_pos = np.sqrt(sigma_r**2 + (r_approx * sigma_b)**2 + 
                                np.trace(self.Sigma[:2, :2]))
            self._candidates[self._next_candidate_id] = {
                'x': x_obs, 'y': y_obs,
                'count': 1,
                'sigma_r': sigma_pos,
                'sigma_b': sigma_b * 3,
            }
            self._next_candidate_id += 1
            return None

    def _get_or_create_landmark(self, gt_id: int,
                                r_obs: float, b_obs: float) -> int:
        """Known-association mode: map ground-truth ID to internal index."""
        if gt_id in self._gt_to_internal:
            j = self._gt_to_internal[gt_id]
            self.observation_count[j] += 1
            return j
        j = self._add_landmark_slot()
        self._gt_to_internal[gt_id] = j
        return j

    # ── LANDMARK MANAGEMENT ─────────────────────────────────────────────────

    def _add_landmark_slot(self) -> int:
        """Expand state vector by 2 for a new landmark. Returns internal index."""
        j = self.n_landmarks
        self.n_landmarks += 1
        self.initialized.append(False)
        self.observation_count.append(0)

        old_n = len(self.mu)
        self._lm_state_idx.append(old_n)

        # Expand mu
        self.mu = np.append(self.mu, [0.0, 0.0])

        # Expand Sigma: new landmark gets large uncertainty; cross-terms = 0
        new_n = old_n + 2
        Sigma_new = np.zeros((new_n, new_n))
        Sigma_new[:old_n, :old_n] = self.Sigma
        Sigma_new[old_n, old_n] = 1e6
        Sigma_new[old_n + 1, old_n + 1] = 1e6
        self.Sigma = Sigma_new

        return j

    def _init_landmark(self, j: int, r_obs: float, b_obs: float):
        """Initialise landmark position from first observation."""
        idx = self._lm_state_idx[j]
        th = self.mu[2]
        self.mu[idx] = self.mu[0] + r_obs * np.cos(b_obs + th)
        self.mu[idx + 1] = self.mu[1] + r_obs * np.sin(b_obs + th)
        self.initialized[j] = True
        self.observation_count[j] = 1

        # Set initial landmark covariance via inverse Jacobian (proper init)
        H = self._landmark_jacobian(idx)
        try:
            H_inv = np.linalg.inv(H)
            self.Sigma[idx:idx+2, idx:idx+2] = H_inv @ self.Q @ H_inv.T
        except np.linalg.LinAlgError:
            pass  # Keep 1e6 diagonal if degenerate

    def _landmark_jacobian(self, idx: int) -> np.ndarray:
        """2x2 Jacobian of observation h w.r.t. landmark position."""
        dx = self.mu[idx] - self.mu[0]
        dy = self.mu[idx + 1] - self.mu[1]
        q = max(dx**2 + dy**2, 1e-6)
        sq = np.sqrt(q)
        return np.array([
            [dx / sq, dy / sq],
            [-dy / q, dx / q],
        ])

    # ── EKF CORRECTION ──────────────────────────────────────────────────────

    def _compute_observation_model(self, idx: int):
        """Compute predicted observation, full Jacobian H, and innovation cov S."""
        dx = self.mu[idx] - self.mu[0]
        dy = self.mu[idx + 1] - self.mu[1]
        q = max(dx**2 + dy**2, 1e-6)
        sq = np.sqrt(q)

        z_hat = np.array([sq, self._wrap(np.arctan2(dy, dx) - self.mu[2])])

        n = len(self.mu)
        H = np.zeros((2, n))
        H[0, 0] = -dx / sq;  H[0, 1] = -dy / sq
        H[1, 0] = dy / q;    H[1, 1] = -dx / q;   H[1, 2] = -1.0
        H[0, idx] = dx / sq;     H[0, idx + 1] = dy / sq
        H[1, idx] = -dy / q;     H[1, idx + 1] = dx / q

        S = H @ self.Sigma @ H.T + self.Q
        return z_hat, H, S

    def _ekf_correct(self, j: int, idx: int, r_obs: float, b_obs: float):
        """Single-observation EKF correction with Joseph-form covariance update."""
        z_hat, H, S = self._compute_observation_model(idx)
        n = len(self.mu)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        K = self.Sigma @ H.T @ S_inv

        innovation = np.array([r_obs - z_hat[0],
                               self._wrap(b_obs - z_hat[1])])

        # Snapshot robot pose trace BEFORE update (for loop closure scoring)
        trace_before = float(np.trace(self.Sigma[:3, :3]))

        self.mu += K @ innovation
        self.mu[2] = self._wrap(self.mu[2])

        # Joseph form: (I-KH)Σ(I-KH)ᵀ + KQKᵀ  — numerically stable
        IKH = np.eye(n) - K @ H
        self.Sigma = IKH @ self.Sigma @ IKH.T + K @ self.Q @ K.T

        # Loop closure detection:
        # A "real" loop closure is a significant reduction in the ROBOT pose
        # uncertainty. This only happens when the robot re-observes a landmark
        # that was mapped long ago (when robot uncertainty was high).
        # Criteria:
        #   1. Landmark has been observed many times before (it's well-mapped)
        #   2. The robot pose covariance trace was near its historical maximum
        #      (robot was genuinely uncertain / had drifted)
        #   3. The reduction is substantial (> 20% of trace_before)
        trace_after = float(np.trace(self.Sigma[:3, :3]))
        trace_reduction = trace_before - trace_after

        is_well_mapped = self.observation_count[j] >= 5
        # Was robot close to its maximum uncertainty? (drift before closure)
        near_max = (self._max_robot_trace > 1e-6 and
                    trace_before > 0.5 * self._max_robot_trace)
        substantial = (trace_before > 1e-4 and
                       trace_reduction > 0.20 * trace_before)

        if is_well_mapped and near_max and substantial:
            self.loop_closures.append({
                'landmark': j,
                'trace_before': trace_before,
                'trace_after': trace_after,
                'trace_reduction': trace_reduction,
                'robot_pose': self.mu[:3].copy(),
            })

    # ── UTILITY ──────────────────────────────────────────────────────────────

    def get_landmark_position(self, j: int) -> np.ndarray | None:
        """Return estimated [x, y] of internal landmark j, or None."""
        if j >= self.n_landmarks or not self.initialized[j]:
            return None
        idx = self._lm_state_idx[j]
        return self.mu[idx:idx + 2].copy()

    def get_landmark_covariance(self, j: int) -> np.ndarray | None:
        """Return 2x2 covariance block of internal landmark j, or None."""
        if j >= self.n_landmarks or not self.initialized[j]:
            return None
        idx = self._lm_state_idx[j]
        return self.Sigma[idx:idx + 2, idx:idx + 2].copy()

    def get_landmark_indices(self) -> dict[int, int]:
        """Return dict mapping ground-truth landmark ID -> state vector index."""
        if self.known_association:
            return {gt_id: self._lm_state_idx[j]
                    for gt_id, j in self._gt_to_internal.items()}
        return {j: self._lm_state_idx[j]
                for j in range(self.n_landmarks)
                if self.initialized[j]}

    @staticmethod
    def _wrap(a: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi
