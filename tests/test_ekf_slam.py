"""Tests for the EKFSLAM class."""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ekf_slam import EKFSLAM
from simulation import Simulation


@pytest.fixture
def sim():
    return Simulation(max_range=8.0, motion_noise=(0.05, 0.05, 1.0),
                      obs_noise=(0.15, 2.0))


@pytest.fixture
def ekf(sim):
    return EKFSLAM(Q_obs=sim.Q_obs, R_motion=sim.R_motion,
                   known_association=True)


class TestPredict:
    def test_state_dimension_unchanged(self, ekf):
        assert len(ekf.mu) == 3
        ekf.predict((1.0, 0.0), dt=0.1)
        assert len(ekf.mu) == 3

    def test_covariance_grows(self, ekf):
        trace_before = np.trace(ekf.Sigma[:3, :3])
        ekf.predict((1.0, 0.25), dt=0.1)
        trace_after = np.trace(ekf.Sigma[:3, :3])
        assert trace_after > trace_before

    def test_position_updates(self, ekf):
        ekf.predict((1.0, 0.0), dt=1.0)
        # Should move forward roughly 1m in x
        assert ekf.mu[0] == pytest.approx(1.0, abs=0.01)


class TestUpdate:
    def test_landmark_discovery(self, ekf):
        assert ekf.n_landmarks == 0
        ekf.update([(3.0, 0.5, 0)])
        assert ekf.n_landmarks == 1

    def test_landmark_initialised(self, ekf):
        ekf.update([(3.0, 0.5, 0)])
        assert ekf.initialized[0] is True

    def test_state_vector_grows(self, ekf):
        ekf.update([(3.0, 0.5, 0)])
        assert len(ekf.mu) == 5  # 3 robot + 2 landmark

    def test_multiple_landmarks(self, ekf):
        ekf.update([(3.0, 0.5, 0), (4.0, -0.3, 1)])
        assert ekf.n_landmarks == 2
        assert len(ekf.mu) == 7  # 3 + 2*2

    def test_covariance_shrinks_on_re_observation(self, ekf, sim):
        """Re-observing a landmark should reduce its uncertainty."""
        np.random.seed(42)
        state = np.array([0.0, 0.0, 0.0])
        u = (1.0, 0.25)

        # Run a few steps to build up some landmarks
        for _ in range(50):
            state = sim.move_robot(state, u, dt=0.1)
            ekf.predict(u, dt=0.1)
            obs = sim.observe(state)
            ekf.update(obs)

        # Get covariance of first landmark
        cov = ekf.get_landmark_covariance(0)
        assert cov is not None
        # Should be well below initial 1e6
        assert np.trace(cov) < 1.0


class TestUnknownAssociation:
    def test_discovers_landmarks(self, sim):
        np.random.seed(42)
        ekf = EKFSLAM(Q_obs=sim.Q_obs, R_motion=sim.R_motion,
                      known_association=False)
        state = np.array([0.0, 0.0, 0.0])
        u = (1.0, 0.25)
        for _ in range(200):
            state = sim.move_robot(state, u, dt=0.1)
            ekf.predict(u, dt=0.1)
            obs = sim.observe_no_id(state)
            ekf.update(obs)
        # Should find approximately 8 landmarks (the true count)
        assert 6 <= ekf.n_landmarks <= 12


class TestAccessors:
    def test_get_landmark_position(self, ekf):
        ekf.update([(3.0, 0.0, 0)])
        pos = ekf.get_landmark_position(0)
        assert pos is not None
        assert pos.shape == (2,)

    def test_get_nonexistent_landmark(self, ekf):
        pos = ekf.get_landmark_position(99)
        assert pos is None

    def test_get_landmark_indices_known(self, ekf):
        ekf.update([(3.0, 0.0, 0), (4.0, 0.5, 2)])
        indices = ekf.get_landmark_indices()
        assert 0 in indices
        assert 2 in indices


class TestCandidatePruning:
    def test_stale_candidates_pruned(self, sim):
        """Candidates not re-observed within max_candidate_age should be removed."""
        ekf = EKFSLAM(Q_obs=sim.Q_obs, R_motion=sim.R_motion,
                      known_association=False, max_candidate_age=3)
        # Inject one observation that will create a candidate
        ekf.update([(5.0, 0.3)])
        assert len(ekf._candidates) == 1

        # Advance update_count without re-observing that candidate
        for _ in range(4):
            ekf.update([])
        assert len(ekf._candidates) == 0

    def test_active_candidates_kept(self, sim):
        """Candidates that keep being re-observed should survive pruning."""
        ekf = EKFSLAM(Q_obs=sim.Q_obs, R_motion=sim.R_motion,
                      known_association=False, max_candidate_age=5)
        # Create candidate
        ekf.update([(5.0, 0.3)])
        cid = list(ekf._candidates.keys())[0]

        # Re-observe every step to keep it alive
        for _ in range(6):
            ekf.update([(5.0, 0.3)])
            # If it wasn't promoted, check it's still around
            if cid in ekf._candidates:
                assert ekf._candidates[cid]['last_seen'] == ekf._update_count


class TestIntegration:
    def test_full_run_known_association(self, sim):
        """Full simulation run with known association should converge."""
        np.random.seed(42)
        ekf = EKFSLAM(Q_obs=sim.Q_obs, R_motion=sim.R_motion,
                      known_association=True)
        state = np.array([0.0, 0.0, 0.0])
        u = (1.0, 0.25)
        errors = []
        for _ in range(300):
            state = sim.move_robot(state, u, dt=0.1)
            ekf.predict(u, dt=0.1)
            obs = sim.observe(state)
            ekf.update(obs)
            errors.append(np.linalg.norm(state[:2] - ekf.mu[:2]))

        # Trajectory error should stay under 1m
        assert np.mean(errors) < 1.0
        # All 8 landmarks should be discovered
        assert ekf.n_landmarks == 8
