from __future__ import annotations

"""
FastSLAM 1.0 for a 2-D robot with point landmarks.

Each particle carries:
  - A robot pose sample (x, y, theta)  drawn from the motion model
  - A set of per-landmark 2-D EKFs     (independent mean + 2x2 covariance)

Particle weights are maintained in LOG space to avoid underflow/overflow.
Low-variance resampling is triggered when N_eff < M/2.

Data association uses Maximum Likelihood (Mahalanobis) with gating,
with the same candidate-buffer strategy as EKF-SLAM for unknown association.

Reference: Montemerlo, Thrun, Koller, Wegbreit (2002).
           "FastSLAM: A Factored Solution to the Simultaneous
            Localization and Mapping Problem."
"""

import numpy as np
from scipy.stats import chi2

from utils import wrap_angle


class _LandmarkEKF:
    """Per-landmark 2-D EKF (mean and 2x2 covariance)."""
    __slots__ = ('mu', 'Sigma', 'observed', 'obs_count')

    def __init__(self):
        self.mu = np.zeros(2)
        self.Sigma = np.eye(2) * 1e6
        self.observed = False
        self.obs_count = 0


class Particle:
    """Single FastSLAM 1.0 particle."""
    __slots__ = ('pose', 'landmarks', 'log_weight', 'n_landmarks',
                 '_gt_to_internal', '_candidates', '_next_cid')

    def __init__(self):
        self.pose = np.zeros(3)
        self.landmarks: list[_LandmarkEKF] = []
        self.log_weight: float = 0.0   # log of unnormalised weight
        self.n_landmarks: int = 0
        self._gt_to_internal: dict[int, int] = {}
        self._candidates: dict[int, dict] = {}
        self._next_cid: int = 0

    def copy(self) -> 'Particle':
        p = Particle()
        p.pose = self.pose.copy()
        p.log_weight = self.log_weight
        p.n_landmarks = self.n_landmarks
        p._gt_to_internal = dict(self._gt_to_internal)
        p._next_cid = self._next_cid
        p._candidates = {k: dict(v) for k, v in self._candidates.items()}
        p.landmarks = []
        for lm in self.landmarks:
            new_lm = _LandmarkEKF()
            new_lm.mu = lm.mu.copy()
            new_lm.Sigma = lm.Sigma.copy()
            new_lm.observed = lm.observed
            new_lm.obs_count = lm.obs_count
            p.landmarks.append(new_lm)
        return p


class FastSLAM:
    """FastSLAM 1.0 with dynamic landmark management and log-space weights."""

    def __init__(self, n_particles: int, Q_obs: np.ndarray,
                 R_motion: np.ndarray,
                 known_association: bool = False,
                 gate_threshold: float | None = None,
                 min_obs_for_new_lm: int = 2,
                 max_candidate_age: int = 10):
        """
        Parameters
        ----------
        n_particles        : number of particles
        Q_obs              : 2x2 observation noise covariance (range, bearing)
        R_motion           : 3x3 motion noise covariance
        known_association  : if True, observations carry ground-truth landmark IDs
        gate_threshold     : Mahalanobis distance^2 gate (default chi2(2).ppf(0.99))
        min_obs_for_new_lm : candidate confirmations before adding to map
        max_candidate_age  : update calls before an un-matched candidate is discarded
        """
        self.M = n_particles
        self.Q = Q_obs.copy()
        self.R = R_motion.copy()
        self.known_association = known_association
        self.gate_threshold = gate_threshold or chi2.ppf(0.99, df=2)
        self.min_obs = min_obs_for_new_lm
        self.max_candidate_age = max_candidate_age
        self._update_count: int = 0

        self.particles = [Particle() for _ in range(self.M)]

        # Public state (updated after each call to _update_estimate)
        self.mu = np.zeros(3)
        self.Sigma = np.zeros((3, 3))
        self.n_landmarks = 0
        self.initialized: list[bool] = []
        self.loop_closures: list[dict] = []
        self.observation_count: list[int] = []
        self._best_particle_idx: int = 0

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def predict(self, u: tuple[float, float], dt: float = 0.1):
        """Sample new robot poses from the motion model."""
        v, w = u
        for p in self.particles:
            noise = np.random.multivariate_normal(np.zeros(3), self.R)
            th = p.pose[2]
            p.pose[0] += (v + noise[0]) * np.cos(th) * dt
            p.pose[1] += (v + noise[1]) * np.sin(th) * dt
            p.pose[2] = wrap_angle(p.pose[2] + (w + noise[2]) * dt)

    # ── UPDATE ────────────────────────────────────────────────────────────────

    def update(self, observations: list):
        """Update log-weights and per-landmark EKFs for each particle."""
        self._update_count += 1
        for p in self.particles:
            self._prune_stale_candidates(p)

        for obs in observations:
            if self.known_association:
                r_obs, b_obs, gt_id = obs
            else:
                r_obs, b_obs = obs[0], obs[1]
                gt_id = None

            for p in self.particles:
                j = self._associate_particle(p, r_obs, b_obs, gt_id)
                if j is None:
                    continue   # still in candidate buffer
                lm = p.landmarks[j]

                if not lm.observed:
                    # Initialise landmark from this observation
                    th = p.pose[2]
                    lm.mu[0] = p.pose[0] + r_obs * np.cos(b_obs + th)
                    lm.mu[1] = p.pose[1] + r_obs * np.sin(b_obs + th)
                    # Proper covariance init via inverse Jacobian
                    H = _obs_jacobian_landmark(p.pose, lm.mu)
                    try:
                        H_inv_Q = np.linalg.solve(H, self.Q)
                        lm.Sigma = np.linalg.solve(H, H_inv_Q.T).T
                    except np.linalg.LinAlgError:
                        lm.Sigma = np.eye(2) * 1.0
                    lm.observed = True
                    lm.obs_count = 1
                    # New-landmark branch: weight unchanged (uniform importance)
                else:
                    # EKF update + weight increment (log-space)
                    z_hat, H_lm = _predicted_obs(p.pose, lm.mu)
                    S = H_lm @ lm.Sigma @ H_lm.T + self.Q
                    innovation = np.array([
                        r_obs - z_hat[0],
                        wrap_angle(b_obs - z_hat[1])
                    ])

                    try:
                        K = np.linalg.solve(S.T, (lm.Sigma @ H_lm.T).T).T
                        maha2 = float(innovation @ np.linalg.solve(S, innovation))
                    except np.linalg.LinAlgError:
                        continue
                    lm.mu = lm.mu + K @ innovation
                    IKH = np.eye(2) - K @ H_lm
                    lm.Sigma = IKH @ lm.Sigma @ IKH.T + K @ self.Q @ K.T
                    # Enforce symmetry to prevent floating-point drift
                    lm.Sigma = (lm.Sigma + lm.Sigma.T) * 0.5
                    lm.obs_count += 1

                    # Log-likelihood: log N(innovation; 0, S)
                    # = -0.5*(d * log(2*pi) + log|S| + inno'S^{-1}inno)
                    sign, log_det_S = np.linalg.slogdet(S)
                    if sign <= 0:
                        continue
                    log_lik = -0.5 * (2 * np.log(2 * np.pi) + log_det_S + maha2)
                    p.log_weight += log_lik

        self._resample()
        self._update_estimate()

    # ── DATA ASSOCIATION (per particle) ───────────────────────────────────────

    def _associate_particle(self, p: Particle, r_obs: float, b_obs: float,
                            gt_id: int | None) -> int | None:
        """Associate observation to a landmark in particle p. Returns slot index or None."""
        if self.known_association and gt_id is not None:
            if gt_id in p._gt_to_internal:
                return p._gt_to_internal[gt_id]
            j = _add_lm(p)
            p._gt_to_internal[gt_id] = j
            return j

        # ML association against confirmed landmarks
        best_j, best_dist = None, self.gate_threshold
        for j, lm in enumerate(p.landmarks):
            if not lm.observed:
                continue
            z_hat, H = _predicted_obs(p.pose, lm.mu)
            S = H @ lm.Sigma @ H.T + self.Q
            inno = np.array([r_obs - z_hat[0], wrap_angle(b_obs - z_hat[1])])
            try:
                d2 = float(inno @ np.linalg.solve(S, inno))
            except np.linalg.LinAlgError:
                continue
            if d2 < best_dist:
                best_dist, best_j = d2, j

        if best_j is not None:
            return best_j

        # Check candidate buffer
        best_cid, best_cdist = None, self.gate_threshold
        for cid, cand in p._candidates.items():
            dx = cand['x'] - p.pose[0]
            dy = cand['y'] - p.pose[1]
            q = max(dx**2 + dy**2, 1e-6)
            sq = np.sqrt(q)
            z_hat_c = np.array([sq, wrap_angle(np.arctan2(dy, dx) - p.pose[2])])
            S_c = np.diag([cand['sr']**2 + self.Q[0, 0],
                           cand['sb']**2 + self.Q[1, 1]])
            inno_c = np.array([r_obs - z_hat_c[0], wrap_angle(b_obs - z_hat_c[1])])
            try:
                d2_c = float(inno_c @ np.linalg.solve(S_c, inno_c))
            except np.linalg.LinAlgError:
                continue
            if d2_c < best_cdist:
                best_cdist, best_cid = d2_c, cid

        sr = np.sqrt(self.Q[0, 0])
        sb = np.sqrt(self.Q[1, 1])
        th = p.pose[2]
        x_obs = p.pose[0] + r_obs * np.cos(b_obs + th)
        y_obs = p.pose[1] + r_obs * np.sin(b_obs + th)

        if best_cid is not None:
            cand = p._candidates[best_cid]
            n = cand['count'] + 1
            cand['x'] += (x_obs - cand['x']) / n
            cand['y'] += (y_obs - cand['y']) / n
            cand['count'] = n
            cand['last_seen'] = self._update_count
            if n >= self.min_obs:
                j = _add_lm(p)
                p.landmarks[j].mu[:] = [cand['x'], cand['y']]
                del p._candidates[best_cid]
                return j
            return None
        else:
            p._candidates[p._next_cid] = {
                'x': x_obs, 'y': y_obs, 'count': 1,
                'sr': sr, 'sb': sb * 3,
                'last_seen': self._update_count,
            }
            p._next_cid += 1
            return None

    # ── CANDIDATE MANAGEMENT ───────────────────────────────────────────────────

    def _prune_stale_candidates(self, p: 'Particle'):
        """Discard candidates in particle p that haven't been seen recently."""
        cutoff = self._update_count - self.max_candidate_age
        stale = [cid for cid, c in p._candidates.items()
                 if c['last_seen'] < cutoff]
        for cid in stale:
            del p._candidates[cid]

    # ── RESAMPLING ─────────────────────────────────────────────────────────────

    def _resample(self):
        """Low-variance resampling in log space."""
        log_w = np.array([p.log_weight for p in self.particles])
        # Shift for numerical stability before exp
        log_w -= log_w.max()
        weights = np.exp(log_w)
        total = weights.sum()
        if total < 1e-300:
            weights = np.ones(self.M) / self.M
        else:
            weights /= total

        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff > self.M * 0.5:
            # Good diversity — just reset log-weights to uniform
            for p in self.particles:
                p.log_weight = 0.0
            return

        # Low-variance resampling (Thrun et al., Algorithm 4.4)
        new_particles = []
        step = 1.0 / self.M
        r = np.random.uniform(0.0, step)
        c = weights[0]
        i = 0
        for m in range(self.M):
            u = r + m * step
            while u > c and i < self.M - 1:
                i += 1
                c += weights[i]
            new_particles.append(self.particles[i].copy())

        self.particles = new_particles
        for p in self.particles:
            p.log_weight = 0.0

    # ── WEIGHTED ESTIMATE ──────────────────────────────────────────────────────

    def _update_estimate(self):
        """
        Compute the best-particle-based estimate.

        The "best particle" is the one with the highest (log) weight
        after the last resample. We use it for landmark positions.
        Robot pose is the weighted mean across all particles.
        """
        log_w = np.array([p.log_weight for p in self.particles])
        log_w -= log_w.max()
        weights = np.exp(log_w)
        total = weights.sum()
        if total < 1e-300:
            weights = np.ones(self.M) / self.M
        else:
            weights /= total

        # Weighted mean robot pose (handle angle wrapping via circular stats)
        x_m = float(np.dot(weights, [p.pose[0] for p in self.particles]))
        y_m = float(np.dot(weights, [p.pose[1] for p in self.particles]))
        sin_m = float(np.dot(weights, [np.sin(p.pose[2]) for p in self.particles]))
        cos_m = float(np.dot(weights, [np.cos(p.pose[2]) for p in self.particles]))
        th_m = float(np.arctan2(sin_m, cos_m))

        # Weighted sample covariance of robot pose for the visualization ellipse
        dx = np.array([p.pose[0] - x_m for p in self.particles])
        dy = np.array([p.pose[1] - y_m for p in self.particles])
        Sxx = float(np.dot(weights, dx * dx))
        Sxy = float(np.dot(weights, dx * dy))
        Syy = float(np.dot(weights, dy * dy))
        # Add a floor so the ellipse is never invisible even when collapsed
        sigma_floor = 0.05 ** 2
        pose_cov = np.array([[max(Sxx, sigma_floor), Sxy],
                             [Sxy, max(Syy, sigma_floor)]])

        # Best particle: highest weight for landmark estimates
        best_idx = int(np.argmax(weights))
        best = self.particles[best_idx]
        self._best_particle_idx = best_idx

        self.n_landmarks = best.n_landmarks
        self.initialized = [lm.observed for lm in best.landmarks]
        self.observation_count = [lm.obs_count for lm in best.landmarks]

        # Build full mu/Sigma in EKF-SLAM-compatible format
        n = 3 + 2 * self.n_landmarks
        full_mu = np.zeros(n)
        full_mu[0], full_mu[1], full_mu[2] = x_m, y_m, th_m

        full_Sigma = np.eye(n) * 1e6
        full_Sigma[:2, :2] = pose_cov

        for j, lm in enumerate(best.landmarks):
            idx = 3 + 2 * j
            if lm.observed:
                full_mu[idx:idx + 2] = lm.mu
                full_Sigma[idx:idx + 2, idx:idx + 2] = lm.Sigma

        self.mu = full_mu
        self.Sigma = full_Sigma

    # ── API COMPATIBILITY WITH EKFSLAM ─────────────────────────────────────────

    def get_landmark_position(self, j: int) -> np.ndarray | None:
        if j >= self.n_landmarks or not self.initialized[j]:
            return None
        idx = 3 + 2 * j
        return self.mu[idx:idx + 2].copy()

    def get_landmark_covariance(self, j: int) -> np.ndarray | None:
        if j >= self.n_landmarks or not self.initialized[j]:
            return None
        idx = 3 + 2 * j
        return self.Sigma[idx:idx + 2, idx:idx + 2].copy()

    def get_landmark_indices(self) -> dict[int, int]:
        best = self.particles[self._best_particle_idx]
        if self.known_association:
            return {gt_id: 3 + 2 * j
                    for gt_id, j in best._gt_to_internal.items()}
        return {j: 3 + 2 * j
                for j in range(self.n_landmarks)
                if self.initialized[j]}


# ── Module-level helpers (used by both Particle and FastSLAM) ──────────────────

def _add_lm(p: Particle) -> int:
    """Append a blank landmark slot to particle p and return its index."""
    j = p.n_landmarks
    p.n_landmarks += 1
    p.landmarks.append(_LandmarkEKF())
    return j


def _predicted_obs(pose: np.ndarray, lm_mu: np.ndarray):
    """Predicted (range, bearing) and 2x2 landmark Jacobian."""
    dx = lm_mu[0] - pose[0]
    dy = lm_mu[1] - pose[1]
    q = max(dx**2 + dy**2, 1e-6)
    sq = np.sqrt(q)
    z_hat = np.array([sq, wrap_angle(np.arctan2(dy, dx) - pose[2])])
    H = np.array([[dx / sq, dy / sq],
                  [-dy / q, dx / q]])
    return z_hat, H


def _obs_jacobian_landmark(pose: np.ndarray, lm_mu: np.ndarray) -> np.ndarray:
    """2x2 Jacobian of observation model w.r.t. landmark position."""
    _, H = _predicted_obs(pose, lm_mu)
    return H