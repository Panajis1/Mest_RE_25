"""Shared pytest fixtures and session-wide configuration.

The autouse fixture below seeds the global RNGs once per test, so test results
are stable regardless of execution order or pytest-xdist sharding. Tests that
need explicit reproducibility should still construct a local
``np.random.default_rng(seed)`` — this fixture only protects against accidental
non-determinism in code that uses the global state (e.g. legacy helpers).
"""

from __future__ import annotations

import random

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _deterministic_global_rng():
    """Seed numpy + stdlib random before each test, regardless of order."""
    np.random.seed(0)
    random.seed(0)
    yield
