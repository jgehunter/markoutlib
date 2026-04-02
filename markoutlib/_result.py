"""MarkoutResult -- the primary return type of compute()."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

from markoutlib._horizons import Horizon, HorizonSet
from markoutlib._stats import (
    _NO_FIT,
    DecayFitResult,
    block_bootstrap_ci,
    fit_exponential_decay,
    newey_west_tstat,
    permutation_test,
    weighted_mean,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_GROUP_KEYS = ["horizon_type", "horizon_value"]


class MarkoutResult:
    """Holds per-trade markout data and provides analysis methods."""

    def __init__(
        self, data: pl.DataFrame, unit: str, perspective: str = "taker"
    ) -> None:
        self._data = data
        self._unit = unit
        self._perspective = perspective

    # ------------------------------------------------------------------
    # curve
    # ------------------------------------------------------------------

    def curve(
        self,
        *,
        by: str | None = None,
        weight: str | None = None,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
        ci_method: str = "percentile",
        lags: int | None = None,
    ) -> pl.DataFrame:
        """Compute markout curve statistics per horizon.

        Args:
            by: Optional column to group by in addition to horizon keys.
            weight: Optional column of trade-level weights for weighted mean.
            n_bootstrap: Number of bootstrap resamples for CI.
            ci_level: Confidence level for CI.
            ci_method: Bootstrap CI method (currently only 'percentile').
            lags: Fixed lag count for Newey-West; None uses Andrews plug-in.

        Returns:
            DataFrame with one row per (horizon_type, horizon_value[, by]) group.

        Raises:
            ValueError: If weight column does not exist in the data.
        """
        if weight is not None and weight not in self._data.columns:
            msg = f"weight column '{weight}' not found in data"
            raise ValueError(msg)

        group_cols = list(_GROUP_KEYS)
        if by is not None:
            group_cols.append(by)

        rows: list[dict] = []
        for keys, group_df in self._data.group_by(group_cols, maintain_order=True):
            markouts = group_df["markout"].drop_nulls().to_numpy()
            n_obs = len(markouts)
            if n_obs == 0:
                continue

            row: dict = {}
            for col, val in zip(group_cols, keys, strict=True):
                row[col] = val

            # Mean (weighted or unweighted).
            if weight is not None:
                w: NDArray[np.floating] = group_df[weight].drop_nulls().to_numpy()
                row["markout_mean"] = weighted_mean(markouts, w)
                ci_weights = w
            else:
                row["markout_mean"] = float(np.mean(markouts))
                ci_weights = None

            # Quantiles (always unweighted).
            row["markout_median"] = float(np.median(markouts))
            row["markout_q25"] = float(np.percentile(markouts, 25))
            row["markout_q75"] = float(np.percentile(markouts, 75))

            # Shape statistics.
            row["skew"] = float(scipy_skew(markouts))
            row["kurtosis"] = float(scipy_kurtosis(markouts))

            # Bootstrap CI.
            ci_lo, ci_hi = block_bootstrap_ci(
                markouts,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                weights=ci_weights,
            )
            row["markout_ci_lower"] = ci_lo
            row["markout_ci_upper"] = ci_hi

            # Newey-West t-stat.
            t_stat, p_value = newey_west_tstat(markouts, lags=lags)
            row["t_stat"] = t_stat
            row["p_value"] = p_value

            row["n_obs"] = n_obs
            rows.append(row)

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # half_life
    # ------------------------------------------------------------------

    def half_life(
        self,
        *,
        by: str | None = None,
    ) -> DecayFitResult | pl.DataFrame:
        """Fit exponential decay to the markout curve.

        Args:
            by: Optional column to compute per-segment half-lives.

        Returns:
            DecayFitResult when by is None; DataFrame of results otherwise.
        """
        if by is not None:
            return self._half_life_by(by)

        # Filter to positive horizons only
        positive = self._data.filter(pl.col("horizon_value") > 0)
        if positive.height < self._data.height:
            has_negative = self._data.filter(pl.col("horizon_value") < 0).height > 0
            if has_negative:
                import warnings

                warnings.warn(
                    "Negative horizons excluded from decay fit — "
                    "half_life() operates on post-trade horizons only",
                    UserWarning,
                    stacklevel=2,
                )

        horizon_types = positive["horizon_type"].unique()
        if len(horizon_types) != 1:
            return _NO_FIT

        if positive.height == 0:
            return _NO_FIT

        agg = positive.group_by("horizon_value", maintain_order=True).agg(
            pl.col("markout").mean()
        )
        horizons = agg["horizon_value"].to_numpy().astype(float)
        markouts = agg["markout"].to_numpy().astype(float)

        return fit_exponential_decay(horizons, markouts)

    def _half_life_by(self, by: str) -> pl.DataFrame:
        """Compute half-life per segment of *by* column.

        Args:
            by: Column name to segment on.

        Returns:
            DataFrame with one row per segment.
        """
        rows: list[dict] = []
        for (segment_val,), segment_df in self._data.group_by(
            [by], maintain_order=True
        ):
            sub = MarkoutResult(segment_df, self._unit)
            fit = sub.half_life()
            rows.append(
                {
                    by: segment_val,
                    "half_life": fit.half_life,
                    "time_constant": fit.time_constant,
                    "terminal_markout": fit.terminal_markout,
                    "r_squared": fit.r_squared,
                    "converged": fit.converged,
                }
            )
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # test
    # ------------------------------------------------------------------

    def test(
        self,
        column: str,
        *,
        pairwise: bool = False,
        n_permutations: int = 1000,
    ) -> pl.DataFrame:
        """Permutation test for markout differences across segments.

        Args:
            column: Categorical column to segment on.
            pairwise: If True, test all segment pairs with BH correction.
            n_permutations: Number of permutations per test.

        Returns:
            DataFrame of test results.
        """
        if pairwise:
            return self._test_pairwise(column, n_permutations)

        segments = self._data[column].unique(maintain_order=True).to_list()

        rows: list[dict] = []
        for seg in segments:
            mask = self._data[column] == seg
            seg_markouts = self._data.filter(mask)["markout"].drop_nulls().to_numpy()
            comp_markouts = self._data.filter(~mask)["markout"].drop_nulls().to_numpy()

            diff, p_val = permutation_test(
                seg_markouts, comp_markouts, n_permutations=n_permutations
            )
            rows.append(
                {
                    "segment": seg,
                    "segment_n_obs": len(seg_markouts),
                    "segment_mean": float(np.mean(seg_markouts)),
                    "complement_mean": float(np.mean(comp_markouts)),
                    "test_stat": diff,
                    "test_p_value": p_val,
                }
            )

        return pl.DataFrame(rows)

    def _test_pairwise(self, column: str, n_permutations: int) -> pl.DataFrame:
        """All-pairs permutation test with BH correction.

        Args:
            column: Categorical column to segment on.
            n_permutations: Number of permutations per test.

        Returns:
            DataFrame with pairwise comparisons.
        """
        segments = self._data[column].unique(maintain_order=True).to_list()
        rows: list[dict] = []

        for i, seg_a in enumerate(segments):
            for seg_b in segments[i + 1 :]:
                a_marks = (
                    self._data.filter(pl.col(column) == seg_a)["markout"]
                    .drop_nulls()
                    .to_numpy()
                )
                b_marks = (
                    self._data.filter(pl.col(column) == seg_b)["markout"]
                    .drop_nulls()
                    .to_numpy()
                )
                diff, p_val = permutation_test(
                    a_marks, b_marks, n_permutations=n_permutations
                )
                rows.append(
                    {
                        "segment_a": seg_a,
                        "segment_b": seg_b,
                        "diff_mean": diff,
                        "test_stat": diff,
                        "test_p_value_raw": p_val,
                    }
                )

        # Benjamini-Hochberg correction (vectorised).
        m = len(rows)
        if m > 0:
            p_vals = np.array([r["test_p_value_raw"] for r in rows])
            order = np.argsort(p_vals)
            ranks = np.arange(1, m + 1)
            sorted_adjusted = np.minimum(p_vals[order] * m / ranks, 1.0)
            # Enforce monotonicity via reverse cumulative minimum.
            sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
            bh_adjusted = np.empty(m)
            bh_adjusted[order] = sorted_adjusted
            for i, row in enumerate(rows):
                row["test_p_value_bh"] = float(bh_adjusted[i])

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------

    def compare(self, *, weight: str) -> pl.DataFrame:
        """Side-by-side weighted vs unweighted markout means per horizon.

        Args:
            weight: Column of trade-level weights.

        Returns:
            DataFrame with unweighted and weighted means per horizon.

        Raises:
            ValueError: If weight column does not exist in the data.
        """
        if weight not in self._data.columns:
            msg = f"weight column '{weight}' not found in data"
            raise ValueError(msg)

        rows: list[dict] = []
        for keys, group_df in self._data.group_by(_GROUP_KEYS, maintain_order=True):
            valid = group_df.filter(
                pl.col("markout").is_not_null() & pl.col(weight).is_not_null()
            )
            n_obs = len(valid)
            if n_obs == 0:
                continue

            markouts = valid["markout"].to_numpy()
            w = valid[weight].to_numpy()

            row: dict = {}
            for col, val in zip(_GROUP_KEYS, keys, strict=True):
                row[col] = val
            row["markout_unweighted"] = float(np.mean(markouts))
            row["markout_weighted"] = weighted_mean(markouts, w)
            row["n_obs"] = n_obs
            rows.append(row)

        return pl.DataFrame(rows)

    # ------------------------------------------------------------------
    # spread decomposition helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_horizon(horizon: Horizon | HorizonSet) -> tuple[str, float]:
        """Convert a Horizon or single-element HorizonSet to (type_str, value).

        Args:
            horizon: A Horizon or a single-element HorizonSet.

        Returns:
            Tuple of (horizon type string, horizon value).

        Raises:
            ValueError: If a HorizonSet has more than one element.
        """
        if isinstance(horizon, HorizonSet):
            horizon = horizon.single()
        return horizon.type.value, float(horizon.value)

    def _validate_horizon_exists(self, h_type: str, h_value: float) -> None:
        """Check that the given horizon exists in the result data.

        Args:
            h_type: Horizon type string (e.g. 'wall').
            h_value: Horizon value (e.g. 5.0).

        Raises:
            ValueError: If the horizon is not present, listing available ones.
        """
        mask = (self._data["horizon_type"] == h_type) & (
            self._data["horizon_value"] == h_value
        )
        if mask.sum() > 0:
            return
        available: dict[str, list[float]] = {}
        for row in (
            self._data.select("horizon_type", "horizon_value")
            .unique()
            .sort("horizon_type", "horizon_value")
            .iter_rows()
        ):
            available.setdefault(row[0], []).append(row[1])
        parts = [f"{k}={v}" for k, v in sorted(available.items())]
        msg = (
            f"horizon {h_type}({h_value}) not found in result. "
            f"Available: {', '.join(parts)}"
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # effective_spread
    # ------------------------------------------------------------------

    def effective_spread(self, *, by: str | None = None) -> pl.DataFrame:
        """Compute effective spread: side * (price - mid) in bps or price.

        Horizon-independent; deduplicates rows across horizons before computing.

        Args:
            by: Optional column to group by.

        Returns:
            DataFrame with effective_spread_mean, effective_spread_median, n_obs.
        """
        drop_cols = [
            c
            for c in ("horizon_type", "horizon_value", "markout", "future_mid")
            if c in self._data.columns
        ]
        unique_trades = self._data.drop(drop_cols).unique()

        if self._unit == "bps":
            spread_expr = (
                pl.col("side")
                * (pl.col("price") - pl.col("mid"))
                / pl.col("mid")
                * 10_000
            )
        else:
            spread_expr = pl.col("side") * (pl.col("price") - pl.col("mid"))

        trades_with_spread = unique_trades.with_columns(
            spread_expr.alias("_eff_spread")
        )

        group_cols = [by] if by is not None else []
        if group_cols:
            grouped = trades_with_spread.group_by(group_cols, maintain_order=True)
        else:
            grouped = trades_with_spread.group_by(pl.lit(True), maintain_order=True)

        result = grouped.agg(
            pl.col("_eff_spread").mean().alias("effective_spread_mean"),
            pl.col("_eff_spread").median().alias("effective_spread_median"),
            pl.col("_eff_spread").count().alias("n_obs"),
        )

        if not group_cols:
            result = result.drop("literal")

        return result

    # ------------------------------------------------------------------
    # realized_spread
    # ------------------------------------------------------------------

    def realized_spread(
        self, *, horizon: Horizon | HorizonSet, by: str | None = None
    ) -> pl.DataFrame:
        """Compute realized spread: side * (price - future_mid) / mid * 10_000 (bps).

        Args:
            horizon: The horizon to compute realized spread at.
            by: Optional column to group by.

        Returns:
            DataFrame with horizon_type, horizon_value, realized_spread_mean,
            realized_spread_median, n_obs.
        """
        h_type, h_value = self._resolve_horizon(horizon)
        self._validate_horizon_exists(h_type, h_value)

        filtered = self._data.filter(
            (pl.col("horizon_type") == h_type) & (pl.col("horizon_value") == h_value)
        )

        if self._unit == "bps":
            spread_expr = (
                pl.col("side")
                * (pl.col("price") - pl.col("future_mid"))
                / pl.col("mid")
                * 10_000
            )
        else:
            spread_expr = pl.col("side") * (pl.col("price") - pl.col("future_mid"))

        filtered = filtered.with_columns(spread_expr.alias("_real_spread"))

        group_cols = [by] if by is not None else []
        if group_cols:
            grouped = filtered.group_by(group_cols, maintain_order=True)
        else:
            grouped = filtered.group_by(pl.lit(True), maintain_order=True)

        result = grouped.agg(
            pl.col("_real_spread").mean().alias("realized_spread_mean"),
            pl.col("_real_spread").median().alias("realized_spread_median"),
            pl.col("_real_spread").count().alias("n_obs"),
        )

        if not group_cols:
            result = result.drop("literal")

        result = result.with_columns(
            pl.lit(h_type).alias("horizon_type"),
            pl.lit(h_value).alias("horizon_value"),
        )

        # Reorder so horizon cols come first.
        cols = ["horizon_type", "horizon_value"] + [
            c for c in result.columns if c not in ("horizon_type", "horizon_value")
        ]
        return result.select(cols)

    # ------------------------------------------------------------------
    # price_impact
    # ------------------------------------------------------------------

    def price_impact(
        self, *, horizon: Horizon | HorizonSet, by: str | None = None
    ) -> pl.DataFrame:
        """Compute price impact: side * (future_mid - mid) / mid * 10_000 (bps).

        Args:
            horizon: The horizon to compute price impact at.
            by: Optional column to group by.

        Returns:
            DataFrame with horizon_type, horizon_value, price_impact_mean,
            price_impact_median, n_obs.
        """
        h_type, h_value = self._resolve_horizon(horizon)
        self._validate_horizon_exists(h_type, h_value)

        filtered = self._data.filter(
            (pl.col("horizon_type") == h_type) & (pl.col("horizon_value") == h_value)
        )

        if self._unit == "bps":
            impact_expr = (
                pl.col("side")
                * (pl.col("future_mid") - pl.col("mid"))
                / pl.col("mid")
                * 10_000
            )
        else:
            impact_expr = pl.col("side") * (pl.col("future_mid") - pl.col("mid"))

        filtered = filtered.with_columns(impact_expr.alias("_price_impact"))

        group_cols = [by] if by is not None else []
        if group_cols:
            grouped = filtered.group_by(group_cols, maintain_order=True)
        else:
            grouped = filtered.group_by(pl.lit(True), maintain_order=True)

        result = grouped.agg(
            pl.col("_price_impact").mean().alias("price_impact_mean"),
            pl.col("_price_impact").median().alias("price_impact_median"),
            pl.col("_price_impact").count().alias("n_obs"),
        )

        if not group_cols:
            result = result.drop("literal")

        result = result.with_columns(
            pl.lit(h_type).alias("horizon_type"),
            pl.lit(h_value).alias("horizon_value"),
        )

        cols = ["horizon_type", "horizon_value"] + [
            c for c in result.columns if c not in ("horizon_type", "horizon_value")
        ]
        return result.select(cols)

    # ------------------------------------------------------------------
    # spread_decomposition
    # ------------------------------------------------------------------

    def spread_decomposition(
        self, *, horizon: Horizon | HorizonSet, by: str | None = None
    ) -> pl.DataFrame:
        """Decompose effective spread into realized spread + price impact.

        Identity: effective_spread = realized_spread + price_impact (exact).

        Args:
            horizon: The horizon for realized spread and price impact.
            by: Optional column to group by.

        Returns:
            DataFrame with effective_spread_mean, realized_spread_mean,
            price_impact_mean, and n_obs columns.
        """
        eff = self.effective_spread(by=by)
        real = self.realized_spread(horizon=horizon, by=by)
        imp = self.price_impact(horizon=horizon, by=by)

        if by is not None:
            join_on = [by]
            result = eff.join(
                real.select(
                    [
                        by,
                        "horizon_type",
                        "horizon_value",
                        "realized_spread_mean",
                        "realized_spread_median",
                    ]
                ),
                on=join_on,
            ).join(
                imp.select([by, "price_impact_mean", "price_impact_median"]),
                on=join_on,
            )
        else:
            result = eff.with_columns(
                real["horizon_type"],
                real["horizon_value"],
                real["realized_spread_mean"],
                real["realized_spread_median"],
                imp["price_impact_mean"],
                imp["price_impact_median"],
            )

        cols = (["horizon_type", "horizon_value"] + ([by] if by else [])) + [
            "effective_spread_mean",
            "effective_spread_median",
            "realized_spread_mean",
            "realized_spread_median",
            "price_impact_mean",
            "price_impact_median",
            "n_obs",
        ]
        return result.select([c for c in cols if c in result.columns])

    # ------------------------------------------------------------------
    # plot (lazy accessor, wired in Task 12)
    # ------------------------------------------------------------------

    @property
    def plot(self):
        """Return a PlotAccessor for this result."""
        from markoutlib.viz import PlotAccessor

        return PlotAccessor(self)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def to_polars(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame."""
        return self._data

    def to_pandas(self):
        """Return data as a pandas DataFrame."""
        from markoutlib._compat import to_pandas

        return to_pandas(self._data)
