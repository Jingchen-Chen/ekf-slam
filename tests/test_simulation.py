"""Tests for the Simulation class."""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation import Simulation


class TestSimulationInit:
    def test_default_landmarks(self):
        sim = Simulation()
        assert sim.landmarks.shape == (8, 2)

    def test_custom_landmarks(self):
        lm = np.array([[1, 2], [3, 4], [5, 6]])
        sim = Simulation(landmarks=lm)
        assert sim.landmarks.shape == (3, 2)
        np.testing.assert_array_equal(sim.landmarks, lm)

    def test_noise_covariances(self):
        sim = Simulation()
        assert sim.R_motion.shape == (3, 3)
        assert sim.Q_obs.shape == (2, 2)
        # Covariance matrices must be symmetric positive semi-definite
        np.testing.assert_array_equal(sim.R_motion, sim.R_motion.T)
        np.testing.assert_array_equal(sim.Q_obs, sim.Q_obs.T)


class TestMoveRobot:
    def test_returns_3d_state(self):
        sim = Simulation()
        state = np.array([0.0, 0.0, 0.0])
        new_state = sim.move_robot(state, (1.0, 0.0), dt=0.1)
        assert new_state.shape == (3,)

    def test_straight_line_mean(self):
        """With zero noise, robot should move forward along x."""
        sim = Simulation(motion_noise=(0.0, 0.0, 0.0))
        state = np.array([0.0, 0.0, 0.0])
        new_state = sim.move_robot(state, (1.0, 0.0), dt=1.0)
        assert new_state[0] == pytest.approx(1.0, abs=0.01)
        assert new_state[1] == pytest.approx(0.0, abs=0.01)

    def test_angle_wrapping(self):
        """Angle should stay in [-pi, pi]."""
        sim = Simulation(motion_noise=(0.0, 0.0, 0.0))
        state = np.array([0.0, 0.0, 3.0])
        # omega = 1 rad/s, dt = 1s -> theta should wrap
        new_state = sim.move_robot(state, (0.0, 1.0), dt=1.0)
        assert -math.pi <= new_state[2] <= math.pi


class TestObserve:
    def test_known_id_observations(self):
        lm = np.array([[1.0, 0.0], [0.0, 1.0]])
        sim = Simulation(landmarks=lm, max_range=5.0,
                         obs_noise=(0.0, 0.0))
        state = np.array([0.0, 0.0, 0.0])
        obs = sim.observe(state)
        # Both landmarks should be in range
        assert len(obs) == 2
        # Each observation should be (range, bearing, id)
        for r, b, lid in obs:
            assert lid in (0, 1)
            assert r > 0

    def test_no_id_observations(self):
        lm = np.array([[1.0, 0.0]])
        sim = Simulation(landmarks=lm, max_range=5.0,
                         obs_noise=(0.0, 0.0))
        state = np.array([0.0, 0.0, 0.0])
        obs = sim.observe_no_id(state)
        assert len(obs) == 1
        assert len(obs[0]) == 2  # (range, bearing) only, no id

    def test_out_of_range(self):
        lm = np.array([[100.0, 100.0]])
        sim = Simulation(landmarks=lm, max_range=5.0)
        state = np.array([0.0, 0.0, 0.0])
        obs = sim.observe(state)
        assert len(obs) == 0
