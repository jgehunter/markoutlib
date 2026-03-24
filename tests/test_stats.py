"""Tests for statistical functions."""

import numpy as np
import pytest


def test_weighted_mean_basic():
    from markoutlib._stats import weighted_mean

    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    assert weighted_mean(values, weights) == pytest.approx(2.0)


def test_weighted_mean_unequal():
    from markoutlib._stats import weighted_mean

    values = np.array([1.0, 3.0])
    weights = np.array([3.0, 1.0])
    assert weighted_mean(values, weights) == pytest.approx(1.5)


def test_newey_west_zero_mean():
    from markoutlib._stats import newey_west_tstat

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)
    t, p = newey_west_tstat(data)
    assert p > 0.01


def test_newey_west_large_mean():
    from markoutlib._stats import newey_west_tstat

    data = np.ones(1000) * 5.0 + np.random.default_rng(42).normal(0, 0.1, 1000)
    t, p = newey_west_tstat(data)
    assert p < 0.001
    assert t > 10


def test_newey_west_with_lags():
    from markoutlib._stats import newey_west_tstat

    data = np.ones(100)
    t, p = newey_west_tstat(data, lags=5)
    assert p < 0.001


def test_block_bootstrap_ci_covers_true_mean():
    from markoutlib._stats import block_bootstrap_ci

    rng = np.random.default_rng(42)
    data = rng.normal(5.0, 1.0, 500)
    lower, upper = block_bootstrap_ci(data, n_bootstrap=500, ci_level=0.95, seed=42)
    assert lower < 5.0 < upper


def test_block_bootstrap_ci_null_for_small_n():
    from markoutlib._stats import block_bootstrap_ci

    data = np.array([1.0, 2.0, 3.0])
    lower, upper = block_bootstrap_ci(data, n_bootstrap=100, ci_level=0.95)
    assert lower is None
    assert upper is None


def test_permutation_test_different_groups():
    from markoutlib._stats import permutation_test

    rng = np.random.default_rng(42)
    segment = rng.normal(5.0, 0.5, 200)
    complement = rng.normal(0.0, 0.5, 800)
    stat, p = permutation_test(segment, complement, n_permutations=1000, seed=42)
    assert p < 0.01


def test_permutation_test_same_distribution():
    from markoutlib._stats import permutation_test

    rng = np.random.default_rng(42)
    segment = rng.normal(0.0, 1.0, 200)
    complement = rng.normal(0.0, 1.0, 800)
    stat, p = permutation_test(segment, complement, n_permutations=1000, seed=42)
    assert p > 0.01


def test_fit_exponential_decay_converges():
    from markoutlib._stats import fit_exponential_decay

    horizons = np.array([1, 2, 5, 10, 20, 50], dtype=float)
    markouts = 2.0 * (1 - np.exp(-horizons / 5.0))
    result = fit_exponential_decay(horizons, markouts)
    assert result.converged
    assert result.terminal_markout == pytest.approx(2.0, rel=0.05)
    assert result.time_constant == pytest.approx(5.0, rel=0.05)
    assert result.half_life == pytest.approx(5.0 * np.log(2), rel=0.05)


def test_fit_exponential_decay_flat_curve():
    from markoutlib._stats import fit_exponential_decay

    horizons = np.array([1, 2, 5, 10, 20], dtype=float)
    markouts = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    result = fit_exponential_decay(horizons, markouts)
    assert not result.converged
