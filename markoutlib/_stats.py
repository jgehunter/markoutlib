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


def _bootstrap_means_prefixsum(
    cumsum_d: NDArray[np.floating],
    cumsum_w: NDArray[np.floating] | None,
    cumsum_dw: NDArray[np.floating] | None,
    n: int,
    n_bootstrap: int,
    p_geom: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Compute bootstrap resample means using prefix sums.

    Generates all blocks for all resamples at once, assigns them to
    resamples via cumulative sizes, and vectorises the prefix-sum
    lookups. Falls back to a scalar loop only for the per-resample
    last-block cap.
    """
    # Over-allocate blocks for all resamples.
    avg_blocks = n * p_geom  # blocks per resample
    total_est = int(avg_blocks * n_bootstrap * 1.4) + n_bootstrap * 5
    all_starts = rng.integers(0, n, size=total_est)
    all_sizes = rng.geometric(p_geom, size=total_est).astype(np.int64)

    # Find resample boundaries in the flat block array.
    flat_cumsum = np.cumsum(all_sizes)
    thresholds = np.arange(1, n_bootstrap + 1, dtype=np.int64) * n
    boundaries = np.searchsorted(flat_cumsum, thresholds, side="left")

    # Ensure we generated enough blocks.
    while boundaries[-1] >= total_est:
        extra = total_est
        all_starts = np.concatenate([all_starts, rng.integers(0, n, size=extra)])
        all_sizes = np.concatenate(
            [all_sizes, rng.geometric(p_geom, size=extra).astype(np.int64)]
        )
        total_est += extra
        flat_cumsum = np.cumsum(all_sizes)
        boundaries = np.searchsorted(flat_cumsum, thresholds, side="left")

    # Cap the last block of each resample so total == n exactly.
    max_blk = int(boundaries[-1])
    sizes_capped = all_sizes[: max_blk + 1].copy()
    last_blocks = boundaries.astype(np.intp)
    used_before = flat_cumsum[last_blocks] - all_sizes[last_blocks]
    resample_start_totals = np.arange(n_bootstrap, dtype=np.int64) * n
    sizes_capped[last_blocks] = n - (used_before - resample_start_totals)

    starts = all_starts[: max_blk + 1]
    ends = starts + sizes_capped  # may exceed n (circular wrap)

    # Vectorised prefix-sum lookups for all blocks.
    no_wrap = ends <= n
    block_sums = np.empty(len(starts))
    # Non-wrapping blocks
    if np.any(no_wrap):
        block_sums[no_wrap] = cumsum_d[ends[no_wrap]] - cumsum_d[starts[no_wrap]]
    # Wrapping blocks
    wrap = ~no_wrap
    if np.any(wrap):
        wrap_ends = ends[wrap] - n
        block_sums[wrap] = (cumsum_d[n] - cumsum_d[starts[wrap]]) + cumsum_d[wrap_ends]

    # Sum block sums per resample using reduceat.
    resample_starts = np.empty(n_bootstrap, dtype=np.intp)
    resample_starts[0] = 0
    resample_starts[1:] = boundaries[:-1] + 1
    means = np.add.reduceat(block_sums, resample_starts) / n

    if cumsum_w is not None and cumsum_dw is not None:
        # Weighted means: repeat for data*weights and weights.
        bsums_dw = np.empty(len(starts))
        bsums_dw[no_wrap] = cumsum_dw[ends[no_wrap]] - cumsum_dw[starts[no_wrap]]
        if np.any(wrap):
            bsums_dw[wrap] = (cumsum_dw[n] - cumsum_dw[starts[wrap]]) + cumsum_dw[
                ends[wrap] - n
            ]
        bsums_w = np.empty(len(starts))
        bsums_w[no_wrap] = cumsum_w[ends[no_wrap]] - cumsum_w[starts[no_wrap]]
        if np.any(wrap):
            bsums_w[wrap] = (cumsum_w[n] - cumsum_w[starts[wrap]]) + cumsum_w[
                ends[wrap] - n
            ]
        total_dw = np.add.reduceat(bsums_dw, resample_starts)
        total_w = np.add.reduceat(bsums_w, resample_starts)
        means = np.where(total_w > 0, total_dw / total_w, 0.0)

    return means


def block_bootstrap_ci(
    data: NDArray[np.floating],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    weights: NDArray[np.floating] | None = None,
    seed: int | None = None,
) -> tuple[float | None, float | None]:
    """Stationary block bootstrap confidence interval for the mean.

    Uses geometric block lengths (Politis-Romano stationary bootstrap)
    with circular wrapping. Computes block sums via prefix sums to
    avoid materializing full index arrays.

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

    # Prefix sums for O(1) block-sum lookups (avoids 860K-element fancy indexing).
    cumsum_d = np.empty(n + 1)
    cumsum_d[0] = 0.0
    np.cumsum(data, out=cumsum_d[1:])

    if weights is not None:
        dw = data * weights
        cumsum_dw = np.empty(n + 1)
        cumsum_dw[0] = 0.0
        np.cumsum(dw, out=cumsum_dw[1:])
        cumsum_w = np.empty(n + 1)
        cumsum_w[0] = 0.0
        np.cumsum(weights, out=cumsum_w[1:])

    means = _bootstrap_means_prefixsum(
        cumsum_d,
        cumsum_w if weights is not None else None,
        cumsum_dw if weights is not None else None,
        n,
        n_bootstrap,
        p_geom,
        rng,
    )

    alpha = 1.0 - ci_level
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


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
    n_total = len(segment) + len(complement)
    combined = np.concatenate([segment, complement])
    observed_diff = float(np.mean(segment) - np.mean(complement))

    # Vectorised: generate all permutation indices at once.
    base = np.tile(np.arange(n_total), (n_permutations, 1))
    perm_indices = rng.permuted(base, axis=1)
    permuted = combined[perm_indices]  # (n_permutations, n_total)
    seg_means = permuted[:, :n_seg].mean(axis=1)
    comp_means = permuted[:, n_seg:].mean(axis=1)
    perm_diffs = seg_means - comp_means
    count = int(np.sum(np.abs(perm_diffs) >= abs(observed_diff)))

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
