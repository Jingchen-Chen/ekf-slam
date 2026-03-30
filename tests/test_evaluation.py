"""Tests for the SLAMEvaluator class."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation import SLAMEvaluator


class TestTrajectoryMetrics:
    def test_zero_error(self):
        lm = np.array([[1, 2], [3, 4]])
        ev = SLAMEvaluator(lm)
        for _ in range(10):
            pose = np.array([1.0, 2.0])
            ev.record(pose, pose)
        assert ev.trajectory_rmse() == pytest.approx(0.0)

    def test_known_error(self):
        lm = np.array([[0, 0]])
        ev = SLAMEvaluator(lm)
        ev.record(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        assert ev.trajectory_rmse() == pytest.approx(1.0)

    def test_trajectory_errors_shape(self):
        lm = np.array([[0, 0]])
        ev = SLAMEvaluator(lm)
        for _ in range(5):
            ev.record(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
        errors = ev.trajectory_errors()
        assert errors.shape == (5,)


class TestLandmarkErrors:
    def test_known_association(self):
        true_lm = np.array([[1.0, 0.0], [0.0, 1.0]])
        ev = SLAMEvaluator(true_lm)

        # State vector: [x, y, th, lx0, ly0, lx1, ly1]
        mu = np.array([0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 1.1])
        initialized = [True, True]
        indices = {0: 3, 1: 5}

        result = ev.landmark_errors(mu, initialized, indices,
                                    known_association=True)
        assert len(result['per_landmark']) == 2
        assert result['mean_error'] == pytest.approx(0.1, abs=0.01)

    def test_no_landmarks(self):
        ev = SLAMEvaluator(np.array([[0, 0]]))
        result = ev.landmark_errors(np.zeros(3), [], {})
        assert np.isnan(result['mean_error'])


class TestHeadingRMSE:
    def test_zero_heading_error(self):
        lm = np.array([[1, 2]])
        ev = SLAMEvaluator(lm)
        for _ in range(5):
            ev.record(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]))
        assert ev.heading_rmse() == pytest.approx(0.0)

    def test_known_heading_error(self):
        import math
        lm = np.array([[1, 2]])
        ev = SLAMEvaluator(lm)
        # True heading 0, estimated heading pi/4 -- error is pi/4 rad
        ev.record(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, math.pi / 4]))
        assert ev.heading_rmse() == pytest.approx(math.pi / 4, abs=1e-9)

    def test_angle_wrapping(self):
        import math
        lm = np.array([[1, 2]])
        ev = SLAMEvaluator(lm)
        # True=pi, est=-pi: wrap_angle difference should be ~0
        ev.record(np.array([0.0, 0.0, math.pi]),
                  np.array([0.0, 0.0, -math.pi + 1e-9]))
        assert ev.heading_rmse() == pytest.approx(0.0, abs=1e-6)

    def test_heading_rmse_from_simulation(self):
        from ekf_slam import EKFSLAM
        from simulation import Simulation
        np.random.seed(0)
        sim = Simulation()
        ekf = EKFSLAM(sim.Q_obs, sim.R_motion, known_association=True)
        ev = SLAMEvaluator(sim.landmarks)
        state = np.zeros(3)
        u = (1.0, 0.25)
        for _ in range(100):
            state = sim.move_robot(state, u, dt=0.1)
            ekf.predict(u, dt=0.1)
            ekf.update(sim.observe(state))
            ev.record(state, ekf.mu[:3])
        rmse = ev.heading_rmse()
        assert isinstance(rmse, float)
        assert 0.0 <= rmse < 0.5  # well within 0.5 rad for a working filter
