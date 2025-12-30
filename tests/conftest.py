"""
Pytest fixtures for NeutroHydro tests.
"""

import numpy as np
import pytest


@pytest.fixture
def random_state():
    """Fixed random state for reproducibility."""
    return 42


@pytest.fixture
def sample_data(random_state):
    """
    Generate synthetic groundwater-like data.

    Returns (X, y) where:
    - X: Ion concentrations (100 samples, 7 ions)
    - y: Log TDS (target)
    """
    rng = np.random.default_rng(random_state)

    n = 100
    p = 7  # Ca, Mg, Na, K, HCO3, Cl, SO4

    # Generate correlated ion data
    # Base concentrations (log-normal)
    base = rng.lognormal(mean=2.0, sigma=0.5, size=(n, p))

    # Add correlation structure
    correlation_matrix = np.array([
        [1.0, 0.7, 0.3, 0.2, 0.6, 0.3, 0.4],
        [0.7, 1.0, 0.2, 0.2, 0.5, 0.2, 0.3],
        [0.3, 0.2, 1.0, 0.6, 0.3, 0.8, 0.4],
        [0.2, 0.2, 0.6, 1.0, 0.2, 0.4, 0.3],
        [0.6, 0.5, 0.3, 0.2, 1.0, 0.2, 0.3],
        [0.3, 0.2, 0.8, 0.4, 0.2, 1.0, 0.5],
        [0.4, 0.3, 0.4, 0.3, 0.3, 0.5, 1.0],
    ])

    L = np.linalg.cholesky(correlation_matrix)
    X = base @ L.T

    # Target: log TDS approximately sum of ions with noise
    log_tds = np.log(X.sum(axis=1)) + rng.normal(0, 0.1, n)

    return X, log_tds


@pytest.fixture
def ion_names():
    """Standard ion names."""
    return ["Ca2+", "Mg2+", "Na+", "K+", "HCO3-", "Cl-", "SO42-"]


@pytest.fixture
def small_data(random_state):
    """Small dataset for quick tests."""
    rng = np.random.default_rng(random_state)
    n, p = 20, 4
    X = rng.lognormal(1.0, 0.5, (n, p))
    y = np.log(X.sum(axis=1)) + rng.normal(0, 0.1, n)
    return X, y


@pytest.fixture
def meq_data(random_state):
    """
    Generate data in meq/L for mineral inference.

    Returns 9-ion data matching STANDARD_IONS order.
    """
    rng = np.random.default_rng(random_state)
    n = 50

    # Ca, Mg, Na, K, HCO3, Cl, SO4, NO3, F (in meq/L)
    concentrations = np.array([
        rng.lognormal(1.5, 0.3, n),  # Ca2+
        rng.lognormal(1.0, 0.3, n),  # Mg2+
        rng.lognormal(1.2, 0.4, n),  # Na+
        rng.lognormal(0.3, 0.2, n),  # K+
        rng.lognormal(1.8, 0.3, n),  # HCO3-
        rng.lognormal(1.0, 0.4, n),  # Cl-
        rng.lognormal(0.8, 0.3, n),  # SO42-
        rng.lognormal(0.1, 0.3, n),  # NO3-
        rng.lognormal(-1.0, 0.5, n), # F-
        rng.lognormal(-2.0, 0.5, n), # Zn2+ (Trace)
        rng.lognormal(-3.0, 0.5, n), # Cd2+ (Trace)
        rng.lognormal(-3.0, 0.5, n), # Pb2+ (Trace)
        rng.lognormal(-1.5, 0.5, n), # B
        rng.lognormal(-2.5, 0.5, n), # Cu2+ (Trace)
        rng.lognormal(-3.0, 0.5, n), # As (Trace)
        rng.lognormal(-3.0, 0.5, n), # Cr (Trace)
        rng.lognormal(-4.0, 0.5, n), # U (Trace)
    ]).T

    return concentrations
