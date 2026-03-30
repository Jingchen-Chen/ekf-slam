"""
Shared utility functions for the SLAM project.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi]. Accepts scalar or ndarray."""
    return (a + np.pi) % (2 * np.pi) - np.pi
