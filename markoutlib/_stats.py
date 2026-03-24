"""Statistical functions for markout analysis.

Pure functions — no state, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.stats import norm

# Minimum sample size for bootstrap CI.
_MIN_BOOTSTRAP_N = 30

# Andrews (1991) AR(1) plug-in constant.
_ANDREWS_CONSTANT = 1.1447


def weighted_mean(values: NDArray[np.floating], weights: NDArray[np.floating]) -> float:
    """Compute weighted arithmetic mean.

    Args:
        values: Array of values.
        weights: Array of non-negative weights, same length as values.

    Returns:
        Weighted mean as a float.
    """
    return float(np.average(values, weights=weights))


def newey_west_tstat(
    data: NDArray[np.floating], lags: int | None = None
) -> tuple[float, float]:
    """Newey-West HAC t-statistic for H0: mean = 0.

    Args:
        data: 1-D array of observations.
        lags: Number of lags for HAC estimator. If None, uses Andrews (1991)
            AR(1) plug-in bandwidth.

    Returns:
        Tuple of (t_statistic, two_sided_p_value).
    """
    n = len(data)
    mean = float(np.mean(data))
    demeaned = data - mean

    if lags is None:
        lags = _andrews_bandwidth(demeaned, n)

    # Autocovariances.
    gamma_0 = float(np.dot(demeaned, demeaned)) / n
    nw_var = gamma_0
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(demeaned[j:], demeaned[:-j])) / n
        bartlett_weight = 1.0 - j / (lags + 1)
        nw_var += 2.0 * bartlett_weight * gamma_j

    se = np.sqrt(nw_var / n)
    t_stat = (np.inf if mean != 0.0 else 0.0) if se == 0.0 else mean / se

    p_value = 2.0 * (1.0 - norm.cdf(abs(t_stat)))
    return float(t_stat), float(p_value)


def _andrews_bandwidth(demeaned: NDArray[np.floating], n: int) -> int:
    """Andrews (1991) AR(1) plug-in bandwidth selection.

    Args:
        demeaned: Mean-centered data.
        n: Sample size.

    Returns:
        Integer number of lags.
    """
    if n < 2:
        return 0
    # AR(1) coefficient via OLS.
    denom = float(np.dot(demeaned[:-1], demeaned[:-1]))
    rho = float(np.dot(demeaned[1:], demeaned[:-1]) / denom) if denom > 0 else 0.0
    rho = max(min(rho, 0.99), -0.99)
    alpha = (4.0 * rho**2) / (1.0 - rho**2) ** 2
    bandwidth = _ANDREWS_CONSTANT * (alpha * n) ** (1.0 / 3.0)
    return min(int(bandwidth), n - 1)


def block_bootstrap_ci(
    data: NDArray[np.floating],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    weights: NDArray[np.floating] | None = None,
    seed: int | None = None,
) -> tuple[float | None, float | None]:
    """Stationary block bootstrap confidence interval for the mean.

    Uses geometric block lengths (Politis-Romano stationary bootstrap)
    with circular wrapping.

    Args:
        data: 1-D array of observations.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g. 0.95 for 95% CI).
        weights: Optional weights for weighted mean per resample.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower, upper) bounds, or (None, None) if n < 30.
    """
    n = len(data)
    if n < _MIN_BOOTSTRAP_N:
        return None, None

    rng = np.random.default_rng(seed)
    block_len = max(int(round(n ** (1.0 / 3.0))), 1)
    p_geom = 1.0 / block_len

    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        indices = _stationary_bootstrap_indices(n, p_geom, rng)
        if weights is not None:
            means[b] = weighted_mean(data[indices], weights[indices])
        else:
            means[b] = float(np.mean(data[indices]))

    alpha = 1.0 - ci_level
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def _stationary_bootstrap_indices(
    n: int, p_geom: float, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Generate stationary bootstrap indices with circular wrapping.

    Args:
        n: Sample size.
        p_geom: Probability parameter for geometric block lengths.
        rng: NumPy random generator.

    Returns:
        Array of n resampled indices.
    """
    indices = np.empty(n, dtype=np.intp)
    i = 0
    while i < n:
        # Start a new block at a random position.
        start = rng.integers(0, n)
        # Geometric block length.
        block_size = rng.geometric(p_geom)
        end = min(i + block_size, n)
        for j in range(i, end):
            indices[j] = (start + (j - i)) % n  # Circular wrap.
        i = end
    return indices


def permutation_test(
    segment: NDArray[np.floating],
    complement: NDArray[np.floating],
    n_permutations: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Two-sample permutation test for difference in means.

    Args:
        segment: First group observations.
        complement: Second group observations.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (observed_difference, p_value).
    """
    rng = np.random.default_rng(seed)
    n_seg = len(segment)
    combined = np.concatenate([segment, complement])
    observed_diff = float(np.mean(segment) - np.mean(complement))

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = float(np.mean(combined[:n_seg]) - np.mean(combined[n_seg:]))
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    # Continuity correction.
    p_value = (count + 1) / (n_permutations + 1)
    return observed_diff, p_value


@dataclass(frozen=True)
class DecayFitResult:
    """Result of exponential decay curve fit.

    Attributes:
        half_life: Time to reach 50% of terminal value.
        time_constant: Exponential time constant (tau).
        terminal_markout: Asymptotic markout level.
        r_squared: Coefficient of determination.
        converged: Whether the fit converged successfully.
    """

    half_life: float | None
    time_constant: float | None
    terminal_markout: float | None
    r_squared: float | None
    converged: bool


_NO_FIT = DecayFitResult(
    half_life=None,
    time_constant=None,
    terminal_markout=None,
    r_squared=None,
    converged=False,
)


def fit_exponential_decay(
    horizons: NDArray[np.floating],
    markouts: NDArray[np.floating],
) -> DecayFitResult:
    """Fit exponential decay model: markout(h) = terminal * (1 - exp(-h / tau)).

    Args:
        horizons: Array of horizon values (positive).
        markouts: Array of corresponding markout values.

    Returns:
        DecayFitResult with fitted parameters or converged=False on failure.
    """
    n = len(horizons)
    if n < 3:
        return _NO_FIT

    # Flat curve check — no signal to fit.
    if np.ptp(markouts) < 1e-12:
        return _NO_FIT

    def _model(
        h: NDArray[np.floating], terminal: float, tau: float
    ) -> NDArray[np.floating]:
        return terminal * (1.0 - np.exp(-h / tau))

    try:
        p0 = [float(markouts[-1]), float(horizons[n // 2])]
        popt, _ = optimize.curve_fit(_model, horizons, markouts, p0=p0, maxfev=5000)
        terminal, tau = float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return _NO_FIT

    if tau <= 0:
        return _NO_FIT

    fitted = _model(horizons, terminal, tau)
    ss_res = float(np.sum((markouts - fitted) ** 2))
    ss_tot = float(np.sum((markouts - np.mean(markouts)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return DecayFitResult(
        half_life=tau * np.log(2),
        time_constant=tau,
        terminal_markout=terminal,
        r_squared=r_squared,
        converged=True,
    )
