"""
Side-by-side comparison of EKF-SLAM and FastSLAM 1.0.

Runs both algorithms on the SAME scenario (identical noise draws via a
shared pre-generated noise sequence) and reports RMSE, landmark errors,
and runtime.

Usage:
    python compare.py
    python compare.py --steps 500 --particles 100 --seed 42
    python compare.py --unknown-association
"""

import argparse
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ekf_slam import EKFSLAM
from fastslam import FastSLAM
from simulation import Simulation
from evaluation import SLAMEvaluator
from utils import wrap_angle
from visualization import plot_trajectory_comparison


def parse_args():
    p = argparse.ArgumentParser(
        description='Compare EKF-SLAM vs FastSLAM 1.0.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--steps', type=int, default=500,
                   help='simulation steps')
    p.add_argument('--dt', type=float, default=0.1,
                   help='time step')
    p.add_argument('--vel', type=float, default=1.0,
                   help='linear velocity (m/s)')
    p.add_argument('--omega', type=float, default=0.25,
                   help='angular velocity (rad/s); 0.25 => R=4m circle')
    p.add_argument('--max-range', type=float, default=8.0,
                   help='sensor max range')
    p.add_argument('--seed', type=int, default=42,
                   help='random seed')
    p.add_argument('--particles', type=int, default=50,
                   help='number of FastSLAM particles')
    p.add_argument('--unknown-association', action='store_true',
                   help='use unknown data association for both')
    p.add_argument('--no-gui', action='store_true',
                   help='headless mode')
    p.add_argument('--plot-dir', type=str, default='.',
                   help='output directory for plots')
    return p.parse_args()


def _pre_generate_noise(sim: Simulation, steps: int,
                        state0: np.ndarray,
                        u: tuple[float, float], dt: float):
    """
    Pre-generate all noise so both algorithms see exactly the same
    ground-truth trajectory and observations.
    """
    motion_noise_seq = []
    obs_noise_seq = []
    state = state0.copy()

    for _ in range(steps):
        m_noise = np.random.multivariate_normal([0, 0, 0], sim.R_motion)
        motion_noise_seq.append(m_noise)

        v, w = u
        x, y, th = state
        x += (v + m_noise[0]) * np.cos(th) * dt
        y += (v + m_noise[1]) * np.sin(th) * dt
        th = wrap_angle(th + (w + m_noise[2]) * dt)
        state = np.array([x, y, th])

        # Generate observation noise for all landmarks
        obs_for_step = []
        for i, (lx, ly) in enumerate(sim.landmarks):
            dx, dy = lx - state[0], ly - state[1]
            r = np.hypot(dx, dy)
            if r < sim.max_range:
                bearing = wrap_angle(np.arctan2(dy, dx) - state[2])
                r_noise = np.random.normal(0, np.sqrt(sim.Q_obs[0, 0]))
                b_noise = np.random.normal(0, np.sqrt(sim.Q_obs[1, 1]))
                obs_for_step.append((i, r, bearing, r_noise, b_noise))
        obs_noise_seq.append(obs_for_step)

    return motion_noise_seq, obs_noise_seq


def _replay_step(state: np.ndarray, u: tuple[float, float],
                 m_noise: np.ndarray, dt: float) -> np.ndarray:
    """Replay one motion step with pre-generated noise."""
    v, w = u
    x, y, th = state
    x += (v + m_noise[0]) * np.cos(th) * dt
    y += (v + m_noise[1]) * np.sin(th) * dt
    th = wrap_angle(th + (w + m_noise[2]) * dt)
    return np.array([x, y, th])


def _replay_obs(obs_data: list, known: bool) -> tuple[list, np.ndarray]:
    """Replay observations from pre-generated noise data."""
    observations = []
    for (lid, r_true, b_true, r_noise, b_noise) in obs_data:
        r_obs = r_true + r_noise
        b_obs = b_true + b_noise
        if known:
            observations.append((r_obs, b_obs, lid))
        else:
            observations.append((r_obs, b_obs))
    return observations


def run_algorithm(algo, sim, u, dt, steps,
                  motion_noise_seq, obs_noise_seq, known):
    """Run one SLAM algorithm with pre-generated noise. Return evaluator + time."""
    evaluator = SLAMEvaluator(sim.landmarks)
    state_true = np.array([0.0, 0.0, 0.0])

    t0 = time.perf_counter()
    for t in range(steps):
        state_true = _replay_step(state_true, u, motion_noise_seq[t], dt)
        algo.predict(u, dt=dt)
        obs = _replay_obs(obs_noise_seq[t], known)
        algo.update(obs)
        evaluator.record(state_true, algo.mu[:3])
    elapsed = time.perf_counter() - t0

    return evaluator, elapsed


def main():
    args = parse_args()

    if args.no_gui:
        matplotlib.use('Agg')

    np.random.seed(args.seed)
    sim = Simulation(max_range=args.max_range)
    known = not args.unknown_association
    u = (args.vel, args.omega)
    state0 = np.array([0.0, 0.0, 0.0])

    print(f"Pre-generating noise for {args.steps} steps (seed={args.seed}) ...")
    motion_noise, obs_noise = _pre_generate_noise(
        sim, args.steps, state0, u, args.dt)

    # -- Run EKF-SLAM ----------------------------------------------------------
    print("\nRunning EKF-SLAM ...")
    np.random.seed(args.seed + 1000)
    ekf = EKFSLAM(
        Q_obs=sim.Q_obs, R_motion=sim.R_motion,
        known_association=known,
    )
    ekf_eval, ekf_time = run_algorithm(
        ekf, sim, u, args.dt, args.steps, motion_noise, obs_noise, known)

    # -- Run FastSLAM 1.0 ------------------------------------------------------
    print(f"Running FastSLAM 1.0 ({args.particles} particles) ...")
    np.random.seed(args.seed + 2000)
    fslam = FastSLAM(
        n_particles=args.particles,
        Q_obs=sim.Q_obs, R_motion=sim.R_motion,
        known_association=known,
    )
    fslam_eval, fslam_time = run_algorithm(
        fslam, sim, u, args.dt, args.steps, motion_noise, obs_noise, known)

    # -- Results ---------------------------------------------------------------
    ekf_lm_idx = ekf.get_landmark_indices()
    fslam_lm_idx = fslam.get_landmark_indices()

    ekf_traj_rmse = ekf_eval.trajectory_rmse()
    fslam_traj_rmse = fslam_eval.trajectory_rmse()

    ekf_lm = ekf_eval.landmark_errors(ekf.mu, ekf.initialized, ekf_lm_idx,
                                       known_association=known)
    fslam_lm = fslam_eval.landmark_errors(fslam.mu, fslam.initialized, fslam_lm_idx,
                                          known_association=known)

    print("\n" + "=" * 65)
    print("  COMPARISON: EKF-SLAM  vs  FastSLAM 1.0")
    print("=" * 65)
    print(f"  {'Metric':<30s}  {'EKF-SLAM':>12s}  {'FastSLAM':>12s}")
    print("-" * 65)
    print(f"  {'Trajectory RMSE (m)':<30s}  {ekf_traj_rmse:>12.4f}  {fslam_traj_rmse:>12.4f}")
    print(f"  {'Landmark RMSE (m)':<30s}  {ekf_lm['rmse']:>12.4f}  {fslam_lm['rmse']:>12.4f}")
    print(f"  {'Landmark mean error (m)':<30s}  {ekf_lm['mean_error']:>12.4f}  {fslam_lm['mean_error']:>12.4f}")
    print(f"  {'Runtime (s)':<30s}  {ekf_time:>12.3f}  {fslam_time:>12.3f}")
    print(f"  {'Landmarks discovered':<30s}  {ekf.n_landmarks:>12d}  {fslam.n_landmarks:>12d}")
    print(f"  {'Loop closures':<30s}  {len(ekf.loop_closures):>12d}  {len(fslam.loop_closures):>12d}")
    print("=" * 65)

    # -- Comparison plots ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: EKF trajectory
    ax = axes[0, 0]
    ekf_true = np.array(ekf_eval.true_poses)
    ekf_est = np.array(ekf_eval.est_poses)
    ax.plot(ekf_true[:, 0], ekf_true[:, 1], 'r-', lw=1.5, alpha=0.7, label='True')
    ax.plot(ekf_est[:, 0], ekf_est[:, 1], 'g--', lw=1.5, alpha=0.7, label='Estimated')
    ax.scatter(sim.landmarks[:, 0], sim.landmarks[:, 1],
               s=100, c='black', marker='*', zorder=5)
    for j in range(ekf.n_landmarks):
        pos = ekf.get_landmark_position(j)
        if pos is not None:
            ax.scatter(pos[0], pos[1], s=60, c='royalblue', marker='^', zorder=4)
    if ekf.loop_closures:
        lc = np.array([lc['robot_pose'][:2] for lc in ekf.loop_closures])
        ax.scatter(lc[:, 0], lc[:, 1], s=150, marker='o',
                   edgecolors='orange', facecolors='none', linewidths=2,
                   zorder=7, label='Loop closure')
    ax.set_aspect('equal')
    ax.set_title(f'EKF-SLAM  (RMSE={ekf_traj_rmse:.3f} m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: FastSLAM trajectory
    ax = axes[0, 1]
    fs_true = np.array(fslam_eval.true_poses)
    fs_est = np.array(fslam_eval.est_poses)
    ax.plot(fs_true[:, 0], fs_true[:, 1], 'r-', lw=1.5, alpha=0.7, label='True')
    ax.plot(fs_est[:, 0], fs_est[:, 1], 'g--', lw=1.5, alpha=0.7, label='Estimated')
    ax.scatter(sim.landmarks[:, 0], sim.landmarks[:, 1],
               s=100, c='black', marker='*', zorder=5)
    for j in range(fslam.n_landmarks):
        pos = fslam.get_landmark_position(j)
        if pos is not None:
            ax.scatter(pos[0], pos[1], s=60, c='royalblue', marker='^', zorder=4)
    ax.set_aspect('equal')
    ax.set_title(f'FastSLAM 1.0  (RMSE={fslam_traj_rmse:.3f} m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: error over time
    ax = axes[1, 0]
    ekf_errors = ekf_eval.trajectory_errors()
    fslam_errors = fslam_eval.trajectory_errors()
    ax.plot(ekf_errors, 'b-', lw=1.0, alpha=0.8, label='EKF-SLAM')
    ax.plot(fslam_errors, 'r-', lw=1.0, alpha=0.8, label='FastSLAM')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Position error (m)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: per-landmark errors
    ax = axes[1, 1]
    ekf_per_lm = ekf_lm['per_landmark']
    fslam_per_lm = fslam_lm['per_landmark']
    if ekf_per_lm and fslam_per_lm:
        ekf_ids = [lid for lid, _ in ekf_per_lm]
        ekf_errs = [e for _, e in ekf_per_lm]
        fslam_dict = {lid: e for lid, e in fslam_per_lm}
        fslam_errs = [fslam_dict.get(lid, 0) for lid in ekf_ids]

        x_pos = np.arange(len(ekf_ids))
        width = 0.35
        ax.bar(x_pos - width / 2, ekf_errs, width, label='EKF-SLAM',
               color='steelblue', alpha=0.8)
        ax.bar(x_pos + width / 2, fslam_errs, width, label='FastSLAM',
               color='coral', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'LM{lid}' for lid in ekf_ids])
        ax.set_ylabel('Position Error (m)')
        ax.set_title('Per-Landmark Position Error')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'EKF-SLAM vs FastSLAM 1.0 | {args.steps} steps | '
                 f'{"unknown" if args.unknown_association else "known"} association',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_path = f'{args.plot_dir}/comparison_ekf_vs_fastslam.png'
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot to '{save_path}'")

    if not args.no_gui:
        plt.show()


if __name__ == '__main__':
    main()
