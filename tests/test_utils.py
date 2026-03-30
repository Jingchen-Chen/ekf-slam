"""Tests for shared utility functions."""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import wrap_angle


class TestWrapAngle:
    def test_zero(self):
        assert wrap_angle(0.0) == pytest.approx(0.0)

    def test_pi(self):
        # pi should map to pi (or -pi, both are equivalent boundaries)
        result = wrap_angle(math.pi)
        assert result == pytest.approx(-math.pi) or result == pytest.approx(math.pi)

    def test_positive_overflow(self):
        # 3*pi wraps to -pi (same angle; both are valid representations of the boundary)
        result = wrap_angle(3 * math.pi)
        assert result == pytest.approx(-math.pi) or result == pytest.approx(math.pi)

    def test_negative(self):
        assert wrap_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)

    def test_negative_overflow(self):
        result = wrap_angle(-3 * math.pi)
        assert -math.pi <= result <= math.pi

    def test_two_pi(self):
        assert wrap_angle(2 * math.pi) == pytest.approx(0.0, abs=1e-10)

    def test_vectorised_input(self):
        """wrap_angle should also work with numpy scalars."""
        result = wrap_angle(np.float64(4.0))
        assert -math.pi < result <= math.pi
