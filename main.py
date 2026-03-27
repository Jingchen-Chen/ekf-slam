"""
EKF-SLAM demo -- 2-D differential-drive robot with point landmarks.

Run:
    python main.py                         # default EKF-SLAM with known association
    python main.py --unknown-association   # ML data association (no landmark IDs)
    python main.py --steps 500 --vel 1.2   # custom parameters
    python main.py --no-gui                # headless, evaluation only
    python main.py --help                  # show all options

Output:
    Real-time matplotlib window + evaluation plots + optional GIF.
"""

import argparse
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ekf_slam import EKFSLAM
from simulation import Simulation
from evaluation import SLAMEvaluator
from visualization import (Visualizer, plot_trajectory_comparison,
                            plot_loop_closure_detail)


def parse_args():
    p = argparse.ArgumentParser(
        description='EKF-SLAM demo with 2-D point landmarks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Simulation parameters
    sim_g = p.add_argument_group('simulation')
    sim_g.add_argument('--steps', type=int, default=500,
                       help='number of simulation steps; '
                            'default 500 = ~2 full circles at omega=0.25')
    sim_g.add_argument('--dt', type=float, default=0.1,
                       help='time step (seconds)')
    sim_g.add_argument('--vel', type=float, default=1.0,
                       help='linear velocity command (m/s)')
    sim_g.add_argument('--omega', type=float, default=0.25,
                       help='angular velocity command (rad/s); '
                            'default 0.25 => R=4m circle, period ~25s')
    sim_g.add_argument('--max-range', type=float, default=8.0,
                       help='sensor maximum range (metres)')
    sim_g.add_argument('--seed', type=int, default=None,
                       help='random seed for reproducibility')

    # Algorithm parameters
    alg_g = p.add_argument_group('algorithm')
    alg_g.add_argument('--unknown-association', action='store_true',
                       help='use ML data association (no known landmark IDs)')
    alg_g.add_argument('--gate-threshold', type=float, default=None,
                       help='Mahalanobis gate threshold for data association')

    # Output parameters
    out_g = p.add_argument_group('output')
    out_g.add_argument('--no-gui', action='store_true',
                       help='disable real-time visualization')
    out_g.add_argument('--gif', type=str, default='ekf_slam_output.gif',
                       help='output GIF path (empty string to skip)')
    out_g.add_argument('--plot-dir', type=str, default='.',
                       help='directory for evaluation plots')
    out_g.add_argument('--render-every', type=int, default=3,
                       help='render every N-th step (for GUI responsiveness)')

    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.no_gui:
        matplotlib.use('Agg')

    # -- Setup -----------------------------------------------------------------
    sim = Simulation(max_range=args.max_range)
    known = not args.unknown_association

    ekf = EKFSLAM(
        Q_obs=sim.Q_obs,
        R_motion=sim.R_motion,
        known_association=known,
        gate_threshold=args.gate_threshold,
    )
    evaluator = SLAMEvaluator(sim.landmarks)

    viz = None
    if not args.no_gui:
        viz = Visualizer(sim.landmarks, record=bool(args.gif))

    state_true = np.array([0.0, 0.0, 0.0])
    u = (args.vel, args.omega)

    assoc_mode = 'known' if known else 'ML (unknown)'
    print(f"Running EKF-SLAM | steps={args.steps} | association={assoc_mode}")

    # -- Main loop -------------------------------------------------------------
    t_start = time.perf_counter()

    for t in range(args.steps):
        state_true = sim.move_robot(state_true, u, dt=args.dt)
        ekf.predict(u, dt=args.dt)

        if known:
            obs = sim.observe(state_true)
        else:
            obs = sim.observe_no_id(state_true)
        ekf.update(obs)

        evaluator.record(state_true, ekf.mu[:3])

        if viz and t % args.render_every == 0:
            # For visualization, always pass obs with potential 3-tuples
            viz_obs = sim.observe(state_true) if not known else obs
            viz.draw(state_true, ekf, viz_obs, label='EKF-SLAM')

    elapsed = time.perf_counter() - t_start

    # -- Evaluation ------------------------------------------------------------
    lm_indices = ekf.get_landmark_indices()
    lm_result = evaluator.landmark_errors(
        ekf.mu, ekf.initialized, lm_indices, known_association=known)
    traj_rmse = evaluator.trajectory_rmse()
    lines = [
        '=' * 55, '  SLAM Evaluation Results', '=' * 55,
        f"  Trajectory RMSE:         {traj_rmse:.4f} m",
        f"  Landmark mean error:     {lm_result['mean_error']:.4f} m",
        f"  Landmark RMSE:           {lm_result['rmse']:.4f} m",
        '-' * 55,
    ]
    for lid, err in lm_result['per_landmark']:
        lines.append(f"    Landmark {lid:2d}:  error = {err:.4f} m")
    lines.append('=' * 55)
    print('\n'.join(lines))
    print(f"  Runtime:                 {elapsed:.3f} s")
    print(f"  Landmarks discovered:    {ekf.n_landmarks}")
    print(f"  Loop closures detected:  {len(ekf.loop_closures)}")

    # -- Trajectory comparison plot --------------------------------------------
    est_landmarks = []
    for j in range(ekf.n_landmarks):
        pos = ekf.get_landmark_position(j)
        cov = ekf.get_landmark_covariance(j)
        if pos is not None:
            est_landmarks.append((pos, cov))

    plot_trajectory_comparison(
        evaluator.true_poses, evaluator.est_poses,
        sim.landmarks, est_landmarks,
        loop_closures=ekf.loop_closures,
        title='EKF-SLAM',
        save_path=f'{args.plot_dir}/ekf_trajectory_comparison.png',
    )

    # -- Loop closure detail ---------------------------------------------------
    if ekf.loop_closures:
        plot_loop_closure_detail(
            ekf.loop_closures,
            evaluator.true_poses, evaluator.est_poses,
            save_path=f'{args.plot_dir}/ekf_loop_closure_detail.png',
        )

    # -- GIF -------------------------------------------------------------------
    if viz and args.gif:
        plt.ioff()
        viz.save_gif(args.gif, fps=25)

    if not args.no_gui:
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
