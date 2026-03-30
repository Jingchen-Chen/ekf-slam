from __future__ import annotations

"""
Visualizer for EKF-SLAM and FastSLAM.

Supports dynamic landmark counts and loop closure event markers.
"""

import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


class Visualizer:
    def __init__(self, landmarks_true: np.ndarray, record: bool = True):
        """
        Parameters
        ----------
        landmarks_true : (N, 2) array  -- ground-truth landmark positions
        record         : bool          -- whether to buffer frames for GIF export
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.tight_layout()

        self.landmarks_true = landmarks_true
        self.record = record
        self._frame_buffers: list[bytes] = []

        # Trajectory history for trail rendering
        self.true_trail: list[np.ndarray] = []
        self.est_trail: list[np.ndarray] = []

        # Loop closure markers
        self.loop_closure_markers: list[np.ndarray] = []

    # -- PUBLIC ----------------------------------------------------------------

    def draw(self, state_true: np.ndarray, ekf, observations: list,
             label: str = 'EKF-SLAM'):
        """Render one frame. Optionally capture it for GIF export."""
        self.true_trail.append(state_true[:2].copy())
        self.est_trail.append(ekf.mu[:2].copy())

        # Check for new loop closures
        n_prev = len(self.loop_closure_markers)
        if hasattr(ekf, 'loop_closures') and len(ekf.loop_closures) > n_prev:
            for lc in ekf.loop_closures[n_prev:]:
                self.loop_closure_markers.append(lc['robot_pose'][:2].copy())

        self._render(state_true, ekf, observations, label)

        if self.record:
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            self._frame_buffers.append(buf.read())
            buf.close()

        plt.pause(0.001)

    def save_gif(self, filename: str = 'ekf_slam.gif', fps: int = 20):
        """Write all buffered frames to an animated GIF."""
        if not self._frame_buffers:
            print("No frames recorded.  Pass record=True to Visualizer.")
            return

        print(f"Saving {len(self._frame_buffers)} frames to '{filename}' ...")
        images = [Image.open(io.BytesIO(b)).convert('RGB')
                  for b in self._frame_buffers]

        duration_ms = int(1000 / fps)
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=duration_ms,
            optimize=False,
        )
        print(f"Saved to '{filename}'.")

    # -- PRIVATE ---------------------------------------------------------------

    def _render(self, state_true, ekf, observations, label):
        ax = self.ax
        ax.cla()

        # Compute axis limits from data
        all_x = [p[0] for p in self.true_trail] + [p[0] for p in self.est_trail]
        all_y = [p[1] for p in self.true_trail] + [p[1] for p in self.est_trail]
        all_x.extend(self.landmarks_true[:, 0])
        all_y.extend(self.landmarks_true[:, 1])
        margin = 3.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} -- 2D Landmark Map', fontsize=14)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        # Trajectory trails
        if len(self.true_trail) > 1:
            trail = np.array(self.true_trail)
            ax.plot(trail[:, 0], trail[:, 1], 'r-', alpha=0.4,
                    linewidth=1.0, label='True path')
        if len(self.est_trail) > 1:
            trail = np.array(self.est_trail)
            ax.plot(trail[:, 0], trail[:, 1], 'g-', alpha=0.4,
                    linewidth=1.0, label='Est. path')

        # Ground-truth landmarks (black stars)
        ax.scatter(self.landmarks_true[:, 0], self.landmarks_true[:, 1],
                   s=150, c='black', marker='*', zorder=5, label='True landmarks')

        # EKF-estimated landmarks + uncertainty ellipses (dynamic)
        first_lm = True
        for j in range(ekf.n_landmarks):
            pos = ekf.get_landmark_position(j)
            if pos is None:
                continue
            cov = ekf.get_landmark_covariance(j)
            ax.scatter(pos[0], pos[1], s=80, c='royalblue', marker='^',
                       zorder=4,
                       label='Est. landmark' if first_lm else None)
            if cov is not None:
                self._draw_ellipse(ax, pos[0], pos[1], cov)
            first_lm = False

        # True robot position
        ax.scatter(*state_true[:2], s=120, c='red', zorder=6, label='True robot')

        # EKF-estimated robot position + uncertainty ellipse
        rx, ry = ekf.mu[0], ekf.mu[1]
        ax.scatter(rx, ry, s=120, c='limegreen', marker='D', zorder=6,
                   label='EKF robot')
        self._draw_ellipse(ax, rx, ry, ekf.Sigma[:2, :2], color='limegreen')

        # Observation rays
        for obs in observations:
            r, b = obs[0], obs[1]
            ex = rx + r * np.cos(b + ekf.mu[2])
            ey = ry + r * np.sin(b + ekf.mu[2])
            ax.plot([rx, ex], [ry, ey], 'y--', alpha=0.35, linewidth=0.8)

        # Loop closure markers
        if self.loop_closure_markers:
            lc_arr = np.array(self.loop_closure_markers)
            ax.scatter(lc_arr[:, 0], lc_arr[:, 1], s=200,
                       marker='o', edgecolors='orange', facecolors='none',
                       linewidths=2, zorder=7, label='Loop closure')

        ax.legend(loc='upper left', fontsize=8)

    @staticmethod
    def _draw_ellipse(ax, cx, cy, cov, color='royalblue', n_std=2):
        """Draw a 2-sigma uncertainty ellipse for a 2x2 covariance block."""
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return
        vals = np.maximum(vals, 0)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * n_std * np.sqrt(vals)

        ell = patches.Ellipse((cx, cy), w, h, angle=angle,
                              color=color, fill=False, linewidth=1.2, alpha=0.7)
        ax.add_patch(ell)


def plot_trajectory_comparison(true_poses: list[np.ndarray],
                               est_poses: list[np.ndarray],
                               landmarks_true: np.ndarray,
                               est_landmarks: list[tuple[np.ndarray, np.ndarray | None]],
                               loop_closures: list[dict] | None = None,
                               title: str = 'EKF-SLAM',
                               save_path: str | None = None):
    """
    Generate a static trajectory comparison figure.

    Parameters
    ----------
    true_poses     : list of [x, y] arrays
    est_poses      : list of [x, y] arrays
    landmarks_true : (N, 2) ground-truth landmark positions
    est_landmarks  : list of (position, covariance_2x2_or_None) per landmark
    loop_closures  : optional list of loop closure event dicts
    title          : figure title
    save_path      : if provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # -- Left panel: trajectory + map --
    ax = axes[0]
    true_arr = np.array(true_poses)
    est_arr = np.array(est_poses)

    ax.plot(true_arr[:, 0], true_arr[:, 1], 'r-', linewidth=1.5,
            label='True trajectory', alpha=0.8)
    ax.plot(est_arr[:, 0], est_arr[:, 1], 'g--', linewidth=1.5,
            label='Estimated trajectory', alpha=0.8)

    ax.scatter(landmarks_true[:, 0], landmarks_true[:, 1],
               s=150, c='black', marker='*', zorder=5, label='True landmarks')

    for pos, cov in est_landmarks:
        ax.scatter(pos[0], pos[1], s=80, c='royalblue', marker='^', zorder=4)
        if cov is not None:
            Visualizer._draw_ellipse(ax, pos[0], pos[1], cov,
                                     color='royalblue', n_std=2)

    if loop_closures:
        lc_poses = np.array([lc['robot_pose'][:2] for lc in loop_closures])
        ax.scatter(lc_poses[:, 0], lc_poses[:, 1], s=200,
                   marker='o', edgecolors='orange', facecolors='none',
                   linewidths=2, zorder=7, label='Loop closure')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{title}: Trajectory & Map')
    ax.legend(fontsize=8)

    # -- Right panel: per-step position error --
    ax2 = axes[1]
    errors = np.sqrt(np.sum((true_arr - est_arr) ** 2, axis=1))
    ax2.plot(errors, 'b-', linewidth=1.0, alpha=0.8)
    ax2.axhline(np.mean(errors), color='r', linestyle='--', linewidth=1.0,
                label=f'Mean = {np.mean(errors):.3f} m')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Position error (m)')
    ax2.set_title(f'{title}: Position Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved trajectory comparison to '{save_path}'")
    return fig


def plot_loop_closure_detail(loop_closures: list[dict],
                             true_poses: list[np.ndarray],
                             est_poses: list[np.ndarray],
                             save_path: str | None = None):
    """
    Visualize loop closure effects: covariance trace reduction over events.
    """
    if not loop_closures:
        print("No loop closure events to visualize.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: trace reduction per event
    ax = axes[0]
    reductions = [lc['trace_reduction'] for lc in loop_closures]
    ax.bar(range(len(reductions)), reductions, color='orange', alpha=0.8,
           edgecolor='darkorange')
    ax.set_xlabel('Loop Closure Event')
    ax.set_ylabel('Covariance Trace Reduction')
    ax.set_title('Covariance Reduction per Loop Closure')
    ax.grid(True, alpha=0.3, axis='y')

    # Right: robot pose trace after each closure
    ax2 = axes[1]
    traces_after = [lc['trace_after'] for lc in loop_closures]
    ax2.plot(traces_after, 'o-', color='steelblue', markersize=5)
    ax2.set_xlabel('Loop Closure Event')
    ax2.set_ylabel('Robot Pose Covariance Trace')
    ax2.set_title('Robot Uncertainty After Loop Closures')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved loop closure detail to '{save_path}'")
    return fig
