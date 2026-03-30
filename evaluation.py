from __future__ import annotations

"""
Quantitative evaluation utilities for SLAM algorithms.

Provides RMSE computation for robot trajectory and landmark position errors.
The landmark matching handles both:
  - Known association: landmark_indices maps gt_id -> state_vector_index.
  - Unknown association: landmark_indices maps slot_id -> state_vector_index;
    evaluation pairs estimated landmarks to ground-truth by nearest-neighbour
    (optimal assignment via Hungarian algorithm for correctness).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import wrap_angle


class SLAMEvaluator:
    """Collects trajectory data and computes evaluation metrics."""

    def __init__(self, landmarks_true: np.ndarray):
        self.landmarks_true = landmarks_true.copy()
        self.true_poses: list[np.ndarray] = []
        self.est_poses: list[np.ndarray] = []
        self.true_headings: list[float] = []
        self.est_headings: list[float] = []

    def record(self, true_pose: np.ndarray, est_pose: np.ndarray):
        """Record one timestep of true and estimated robot pose (x, y, theta)."""
        self.true_poses.append(true_pose[:2].copy())
        self.est_poses.append(est_pose[:2].copy())
        self.true_headings.append(float(true_pose[2]) if len(true_pose) > 2 else 0.0)
        self.est_headings.append(float(est_pose[2]) if len(est_pose) > 2 else 0.0)

    # ── Trajectory metrics ───────────────────────────────────────────────────

    def trajectory_rmse(self) -> float:
        """Root mean square error of the robot position over the full trajectory."""
        true = np.array(self.true_poses)
        est = np.array(self.est_poses)
        return float(np.sqrt(np.mean(np.sum((true - est) ** 2, axis=1))))

    def trajectory_errors(self) -> np.ndarray:
        """Per-step Euclidean position error."""
        true = np.array(self.true_poses)
        est = np.array(self.est_poses)
        return np.sqrt(np.sum((true - est) ** 2, axis=1))

    def heading_rmse(self) -> float:
        """Root mean square error of heading (theta) with proper angle wrapping."""
        errs = np.array([wrap_angle(e - t)
                         for t, e in zip(self.true_headings, self.est_headings)])
        return float(np.sqrt(np.mean(errs ** 2)))

    # ── Landmark metrics ─────────────────────────────────────────────────────

    def landmark_errors(self, mu: np.ndarray, initialized: list[bool],
                        landmark_indices: dict[int, int] | None = None,
                        known_association: bool | None = None) -> dict:
        """
        Compute per-landmark position error with correct gt↔estimate pairing.

        Parameters
        ----------
        mu                : full state vector
        initialized       : list[bool], which landmark slots have been seen
        landmark_indices  : dict from get_landmark_indices():
                              known assoc   -> {gt_id: state_vec_idx}
                              unknown assoc -> {slot_id: state_vec_idx}
        known_association : if None, auto-detects: keys == gt ids when all keys
                            are in range(len(landmarks_true)).

        Strategy
        --------
        - Known association: direct pairing (gt_id from key).
        - Unknown association: extract all estimated positions, then solve
          the assignment problem (Hungarian) to optimally match them to
          ground-truth landmarks (minimum total distance).
        """
        if landmark_indices is None:
            # Legacy fall-through: sequential layout
            landmark_indices = {j: 3 + 2 * j
                                for j in range(len(initialized))
                                if initialized[j]}

        # Collect estimated landmark positions
        est_positions: list[tuple[int, np.ndarray]] = []   # (key, pos)
        for key, idx in landmark_indices.items():
            if len(mu) > idx + 1:
                pos = mu[idx:idx + 2]
                est_positions.append((key, pos))

        if not est_positions:
            return {'per_landmark': [], 'mean_error': float('nan'),
                    'rmse': float('nan'), 'known_association': False}

        n_gt = len(self.landmarks_true)
        keys = [k for k, _ in est_positions]
        est_pos_arr = np.array([p for _, p in est_positions])

        # Determine if keys are ground-truth IDs (known association)
        if known_association is None:
            known_association = all(k in range(n_gt) for k in keys)

        errors = []

        if known_association:
            # Direct pairing: key IS the ground-truth landmark ID
            for key, pos in est_positions:
                if 0 <= key < n_gt:
                    true_pos = self.landmarks_true[key]
                    err = float(np.linalg.norm(pos - true_pos))
                    errors.append((key, err))
        else:
            # Unknown association: Hungarian optimal matching
            # Cost matrix: rows = estimated LMs, cols = gt LMs
            cost = np.full((len(est_positions), n_gt), np.inf)
            for i, (_, epos) in enumerate(est_positions):
                for j, gpos in enumerate(self.landmarks_true):
                    cost[i, j] = np.linalg.norm(epos - gpos)

            # Only keep estimated LMs that are close enough to some gt LM
            # (discard ghost landmarks by threshold = 2 * max_expected_error)
            est_threshold = 2.0  # metres; any match farther is considered ghost

            row_ind, col_ind = linear_sum_assignment(cost)
            gt_matched = set()
            for r, c in zip(row_ind, col_ind):
                dist = cost[r, c]
                if dist < est_threshold and c not in gt_matched:
                    errors.append((c, float(dist)))
                    gt_matched.add(c)

        if not errors:
            return {'per_landmark': [], 'mean_error': float('nan'),
                    'rmse': float('nan'), 'known_association': known_association}

        errs = np.array([e for _, e in errors])
        return {
            'per_landmark': sorted(errors, key=lambda x: x[0]),
            'mean_error': float(np.mean(errs)),
            'rmse': float(np.sqrt(np.mean(errs ** 2))),
            'known_association': known_association,
        }

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self, mu: np.ndarray, initialized: list[bool],
                landmark_indices: dict[int, int] | None = None) -> str:
        """Return a formatted evaluation summary string."""
        traj_rmse = self.trajectory_rmse()
        head_rmse = self.heading_rmse()
        lm = self.landmark_errors(mu, initialized, landmark_indices)

        lines = [
            '=' * 55,
            '  SLAM Evaluation Results',
            '=' * 55,
            f"  Trajectory RMSE:         {traj_rmse:.4f} m",
            f"  Heading RMSE:            {np.degrees(head_rmse):.4f} deg",
            f"  Landmark mean error:     {lm['mean_error']:.4f} m",
            f"  Landmark RMSE:           {lm['rmse']:.4f} m",
            '-' * 55,
        ]
        for lid, err in lm['per_landmark']:
            lines.append(f"    Landmark {lid:2d}:  error = {err:.4f} m")
        lines.append('=' * 55)
        return '\n'.join(lines)
