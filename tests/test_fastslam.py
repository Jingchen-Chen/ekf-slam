"""Tests for the FastSLAM class."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastslam import FastSLAM
from simulation import Simulation


@pytest.fixture
def sim():
    return Simulation(max_range=8.0)


@pytest.fixture
def fslam(sim):
    return FastSLAM(n_particles=10, Q_obs=sim.Q_obs, R_motion=sim.R_motion,
                    known_association=True)


class TestFastSLAMPredict:
    def test_particles_diverge(self, fslam):
        """After predict, particles should have different poses."""
        fslam.predict((1.0, 0.25), dt=0.1)
        poses = np.array([p.pose for p in fslam.particles])
        # Not all poses should be identical
        assert np.std(poses[:, 0]) > 0

    def test_particle_count(self, fslam):
        assert len(fslam.particles) == 10
        fslam.predict((1.0, 0.0), dt=0.1)
        assert len(fslam.particles) == 10


class TestFastSLAMUpdate:
    def test_landmark_discovery(self, fslam):
        fslam.update([(3.0, 0.5, 0)])
        # At least the best particle should have a landmark
        assert fslam.n_landmarks >= 1

    def test_api_compatibility(self, fslam):
        """FastSLAM should provide the same API as EKFSLAM."""
        fslam.update([(3.0, 0.5, 0)])
        pos = fslam.get_landmark_position(0)
        assert pos is not None
        cov = fslam.get_landmark_covariance(0)
        assert cov is not None
        indices = fslam.get_landmark_indices()
        assert isinstance(indices, dict)


class TestFastSLAMIntegration:
    def test_full_run(self, sim):
        """FastSLAM should converge on a full simulation run."""
        np.random.seed(42)
        fslam = FastSLAM(n_particles=30, Q_obs=sim.Q_obs,
                         R_motion=sim.R_motion, known_association=True)
        state = np.array([0.0, 0.0, 0.0])
        u = (1.0, 0.25)
        errors = []
        for _ in range(300):
            state = sim.move_robot(state, u, dt=0.1)
            fslam.predict(u, dt=0.1)
            obs = sim.observe(state)
            fslam.update(obs)
            errors.append(np.linalg.norm(state[:2] - fslam.mu[:2]))

        assert np.mean(errors) < 1.5
        assert fslam.n_landmarks == 8
