"""
EKF-SLAM demo – 2-D differential-drive robot with point landmarks.

Run:
    python main.py

Output:
    Real-time matplotlib window + 'ekf_slam_output.gif' saved after the run.
"""

import numpy as np
import matplotlib.pyplot as plt

from ekf_slam      import EKFSLAM
from simulation    import Simulation
from visualization import Visualizer


def main():
    # ── Setup ─────────────────────────────────────────────────────────────
    sim = Simulation()
    ekf = EKFSLAM(
        n_landmarks=len(sim.landmarks),
        Q_obs=sim.Q_obs,
        R_motion=sim.R_motion,
    )
    viz = Visualizer(sim.landmarks, record=True)

    state_true = np.array([0.0, 0.0, 0.0])

    # ── Main loop – robot drives a large circle ────────────────────────────
    T   = 300
    u   = (1.0, 0.05)   # constant: linear vel = 1 m/s, angular vel = 0.05 rad/s

    for t in range(T):
        # Ground-truth propagation (with noise)
        state_true = sim.move_robot(state_true, u)

        # EKF predict
        ekf.predict(u)

        # Simulate sensor and EKF update
        obs = sim.observe(state_true)
        ekf.update(obs)

        # Render every 3rd step to keep the window responsive
        if t % 3 == 0:
            viz.draw(state_true, ekf, obs)

    print("Simulation complete.")
    plt.ioff()

    # ── Save GIF ──────────────────────────────────────────────────────────
    viz.save_gif('ekf_slam_output.gif', fps=25)

    plt.show()


if __name__ == '__main__':
    main()