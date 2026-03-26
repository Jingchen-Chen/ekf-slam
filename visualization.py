"""
Visualizer for EKF-SLAM.

Performance design
------------------
The previous version called `fig.canvas.renderer.buffer_rgba()` during the
live animation loop, which forced a full rasterisation on every frame even when
the window was not being shown.  This caused a noticeable slowdown.

Fix: keep the live figure for interactive display (lightweight `plt.pause`),
and capture frames separately using `fig.savefig` into an in-memory BytesIO
buffer only when explicitly requested at the end of the run.  Frame capture is
triggered by passing `record=True` to `draw()`.
"""

import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from PIL import Image


class Visualizer:
    def __init__(self, landmarks_true, record: bool = True):
        """
        Parameters
        ----------
        landmarks_true : (N, 2) array  – ground-truth landmark positions
        record         : bool          – whether to buffer frames for GIF export
        """
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.tight_layout()

        self.landmarks_true = landmarks_true
        self.record         = record
        self._frame_buffers: list[bytes] = []   # raw PNG bytes per frame

    # ── PUBLIC ───────────────────────────────────────────────────────────────

    def draw(self, state_true, ekf, observations):
        """Render one frame.  Optionally capture it for GIF export."""
        self._render(state_true, ekf, observations)

        # Capture frame into memory (PNG bytes) – much faster than buffer_rgba
        # because it skips the screen-compositor round-trip.
        if self.record:
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=80)
            buf.seek(0)
            self._frame_buffers.append(buf.read())
            buf.close()

        plt.pause(0.001)

    def save_gif(self, filename: str = 'ekf_slam.gif', fps: int = 20):
        """
        Write all buffered frames to an animated GIF.
        Uses Pillow directly – no dependency on ImageMagick.
        """
        if not self._frame_buffers:
            print("No frames recorded.  Pass record=True to Visualizer.")
            return

        print(f"Saving {len(self._frame_buffers)} frames to '{filename}' …")
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

    # ── PRIVATE ──────────────────────────────────────────────────────────────

    def _render(self, state_true, ekf, observations):
        ax = self.ax
        ax.cla()
        ax.set_xlim(-2, 15)
        ax.set_ylim(-2, 15)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('EKF-SLAM – 2D Landmark Map', fontsize=14)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        # Ground-truth landmarks (black stars)
        ax.scatter(self.landmarks_true[:, 0], self.landmarks_true[:, 1],
                   s=150, c='black', marker='*', zorder=5, label='True landmarks')

        # EKF-estimated landmarks + uncertainty ellipses
        for j in range(len(self.landmarks_true)):
            if not ekf.initialized[j]:
                continue
            idx = 3 + 2 * j
            lx, ly = ekf.mu[idx], ekf.mu[idx + 1]
            ax.scatter(lx, ly, s=80, c='royalblue', marker='^', zorder=4,
                       label='Est. landmark' if j == 0 else None)
            self._draw_ellipse(ax, lx, ly, ekf.Sigma[idx:idx+2, idx:idx+2])

        # True robot position (red circle)
        ax.scatter(*state_true[:2], s=120, c='red', zorder=6, label='True robot')

        # EKF-estimated robot position + uncertainty ellipse (green diamond)
        rx, ry = ekf.mu[0], ekf.mu[1]
        ax.scatter(rx, ry, s=120, c='limegreen', marker='D', zorder=6,
                   label='EKF robot')
        self._draw_ellipse(ax, rx, ry, ekf.Sigma[:2, :2], color='limegreen')

        # Observation rays
        for r, b, _ in observations:
            ex = rx + r * np.cos(b + ekf.mu[2])
            ey = ry + r * np.sin(b + ekf.mu[2])
            ax.plot([rx, ex], [ry, ey], 'y--', alpha=0.35, linewidth=0.8)

        ax.legend(loc='upper left', fontsize=8)

    @staticmethod
    def _draw_ellipse(ax, cx, cy, cov, color='royalblue', n_std=2):
        """Draw a 2-sigma uncertainty ellipse for a 2×2 covariance block."""
        # Guard against near-singular covariances during initialisation
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return
        vals = np.maximum(vals, 0)           # numerical safety: no negative eigenvalues
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h  = 2 * n_std * np.sqrt(vals)

        ell = patches.Ellipse((cx, cy), w, h, angle=angle,
                              color=color, fill=False, linewidth=1.2, alpha=0.7)
        ax.add_patch(ell)