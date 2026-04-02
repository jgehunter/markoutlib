"""Core markout computation engine."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl

from markoutlib._compat import maybe_to_polars
from markoutlib._horizons import Horizon, HorizonSet
from markoutlib._result import MarkoutResult
from markoutlib._types import (
    REQUIRED_QUOTE_COLS,
    REQUIRED_TRADE_COLS,
    HorizonType,
    Unit,
)

try:
    from _markoutlib_native import (
        tick_clock_partition as _tick_clock_partition_rs,
    )
    _USE_NATIVE = True
except ImportError:
    _USE_NATIVE = False


def _validate_inputs(
    trades: pl.DataFrame,
    quotes: pl.DataFrame | None,
    horizons: HorizonSet,
    by: str | list[str] | None,
) -> None:
    for col in REQUIRED_TRADE_COLS:
        if col not in trades.columns:
            msg = f"trades DataFrame missing required column: {col}"
            raise ValueError(msg)

    side = trades["side"]
    if side.null_count() > 0:
        msg = "side column must contain only 1 and -1 (no nulls)"
        raise ValueError(msg)
    unique_sides = set(side.unique().to_list())
    if not unique_sides.issubset({1, -1}):
        msg = "side column must contain only 1 and -1 (no nulls)"
        raise ValueError(msg)

    horizon_types = {h.type for h in horizons}
    needs_quotes = horizon_types & {HorizonType.WALL, HorizonType.TICK}

    if needs_quotes and quotes is None:
        type_name = "wall" if HorizonType.WALL in horizon_types else "tick"
        msg = f"quotes DataFrame required for {type_name} horizons"
        raise ValueError(msg)

    if quotes is not None:
        for col in REQUIRED_QUOTE_COLS:
            if col not in quotes.columns:
                msg = f"quotes DataFrame missing required column: {col}"
                raise ValueError(msg)

        t_dtype = trades["timestamp"].dtype
        q_dtype = quotes["timestamp"].dtype
        if t_dtype != q_dtype:
            msg = f"timestamp column type mismatch: trades={t_dtype}, quotes={q_dtype}"
            raise ValueError(msg)

    if by is not None:
        by_cols = [by] if isinstance(by, str) else by
        for col in by_cols:
            if col not in trades.columns:
                msg = f"by column '{col}' not found in trades DataFrame"
                raise ValueError(msg)
            if quotes is not None and col not in quotes.columns:
                msg = f"by column '{col}' not found in quotes DataFrame"
                raise ValueError(msg)


def _compute_wall_clock(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    horizon: Horizon,
    unit: Unit,
    by: list[str] | None,
) -> pl.DataFrame:
    """Compute markout for a single wall-clock horizon."""
    horizon_seconds = horizon.value
    horizon_td = timedelta(seconds=horizon_seconds)

    # Add target timestamp column to trades
    enriched = trades.with_columns(
        (pl.col("timestamp") + horizon_td).alias("_target_ts"),
    )

    # Prepare quotes for join: rename mid to future_mid
    quote_cols = ["timestamp", "mid"]
    if by is not None:
        quote_cols = ["timestamp", "mid", *by]
    quotes_for_join = quotes.select(quote_cols).rename({"mid": "future_mid"})

    # Backward asof join: find last quote at or before target timestamp
    join_kwargs: dict = {
        "left_on": "_target_ts",
        "right_on": "timestamp",
        "strategy": "backward",
    }
    if by is not None:
        join_kwargs["by"] = by

    joined = enriched.join_asof(quotes_for_join, **join_kwargs)

    # Tolerance filter: if matched quote is more than 2x the horizon away
    # from target, null out future_mid (stale quote protection)
    tolerance_seconds = 2 * abs(horizon_seconds) if horizon_seconds != 0 else 60
    tolerance_td = timedelta(seconds=tolerance_seconds)
    joined = joined.with_columns(
        pl.when(
            (pl.col("timestamp_right").is_null())
            | ((pl.col("_target_ts") - pl.col("timestamp_right")) > tolerance_td)
        )
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("future_mid"))
        .alias("future_mid"),
    )

    # Compute markout
    if unit == Unit.BPS:
        markout_expr = (
            pl.when(
                pl.col("mid").is_null()
                | (pl.col("mid") == 0)
                | pl.col("future_mid").is_null()
            )
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("side").cast(pl.Float64)
                * (pl.col("future_mid") - pl.col("mid"))
                / pl.col("mid")
                * 10_000
            )
        )
    else:
        markout_expr = (
            pl.when(
                pl.col("mid").is_null()
                | (pl.col("mid") == 0)
                | pl.col("future_mid").is_null()
            )
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("side").cast(pl.Float64) * (pl.col("future_mid") - pl.col("mid"))
            )
        )

    joined = joined.with_columns(markout_expr.alias("markout"))

    # Add horizon metadata
    joined = joined.with_columns(
        pl.lit(horizon.type.value).alias("horizon_type"),
        pl.lit(horizon.value).alias("horizon_value"),
    )

    # Drop internal columns
    drop_cols = ["_target_ts"]
    if "timestamp_right" in joined.columns:
        drop_cols.append("timestamp_right")
    joined = joined.drop(drop_cols)

    return joined


def _compute_trade_clock(
    trades: pl.DataFrame,
    horizon: Horizon,
    unit: Unit,
    by_cols: list[str] | None,
) -> pl.DataFrame:
    """Compute markout for a single trade-clock horizon.

    Args:
        trades: DataFrame of trade records with required columns.
        horizon: Horizon specifying how many trades forward to measure.
        unit: Output unit, either BPS or PRICE.
        by_cols: Optional partition columns; shift is applied within each group.

    Returns:
        trades DataFrame with future_mid, markout, horizon_type, and
        horizon_value columns added.
    """
    n = int(horizon.value)

    if by_cols:
        result = trades.with_columns(
            pl.col("mid").shift(-n).over(by_cols).alias("future_mid")
        )
    else:
        result = trades.with_columns(pl.col("mid").shift(-n).alias("future_mid"))

    null_mid = pl.col("mid").is_null() | (pl.col("mid") == 0)

    if unit == Unit.BPS:
        markout_expr = (
            pl.when(null_mid | pl.col("future_mid").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("side").cast(pl.Float64)
                * (pl.col("future_mid") - pl.col("mid"))
                / pl.col("mid")
                * 10_000
            )
        )
    else:
        markout_expr = (
            pl.when(null_mid | pl.col("future_mid").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("side").cast(pl.Float64) * (pl.col("future_mid") - pl.col("mid"))
            )
        )

    return result.with_columns(
        markout_expr.alias("markout"),
        pl.lit(horizon.type.value).alias("horizon_type"),
        pl.lit(horizon.value).alias("horizon_value"),
    )


def _tick_clock_partition_np(
    trade_timestamps: list[int] | np.ndarray,
    quote_timestamps: list[int] | np.ndarray,
    quote_mids: list[float] | np.ndarray,
    n: int,
) -> np.ndarray:
    """Compute tick-clock future mids for one partition (numpy implementation).

    For n > 0: the n-th quote strictly after each trade.
    For n == 0: the last quote at or before each trade.
    For n < 0: the |n|-th quote before the last-at-or-before, counting backward.

    Returns a float64 numpy array with NaN for out-of-bounds lookups.
    """
    quote_ts = np.asarray(quote_timestamps)
    quote_m = np.asarray(quote_mids, dtype=np.float64)
    trade_ts = np.asarray(trade_timestamps)
    num_quotes = len(quote_ts)

    # searchsorted 'right' = first index strictly after each trade (= bisect_right)
    idx = np.searchsorted(quote_ts, trade_ts, side="right")

    if n > 0:
        targets = idx + (n - 1)
    elif n == 0:
        targets = idx - 1
    else:
        targets = (idx - 1) + n  # n is negative

    valid = (targets >= 0) & (targets < num_quotes)
    result = np.full(len(trade_ts), np.nan)
    result[valid] = quote_m[targets[valid]]
    return result


def _tick_clock_partition(
    trade_timestamps: list[int] | np.ndarray,
    quote_timestamps: list[int] | np.ndarray,
    quote_mids: list[float] | np.ndarray,
    n: int,
) -> np.ndarray:
    """Dispatch to Rust or numpy tick-clock implementation."""
    t_ts = np.asarray(trade_timestamps, dtype=np.int64)
    q_ts = np.asarray(quote_timestamps, dtype=np.int64)
    q_m = np.asarray(quote_mids, dtype=np.float64)
    if _USE_NATIVE:
        return np.asarray(_tick_clock_partition_rs(t_ts, q_ts, q_m, n))
    return _tick_clock_partition_np(t_ts, q_ts, q_m, n)


def _compute_tick_clock(
    trades: pl.DataFrame,
    quotes: pl.DataFrame,
    horizon: Horizon,
    unit: Unit,
    by_cols: list[str] | None,
) -> pl.DataFrame:
    """Compute markout for a single tick-clock horizon.

    Args:
        trades: DataFrame of trade records with required columns.
        quotes: DataFrame of quote records, pre-sorted by timestamp.
        horizon: Horizon specifying how many ticks forward to measure.
        unit: Output unit, either BPS or PRICE.
        by_cols: Optional partition columns.

    Returns:
        trades DataFrame with future_mid, markout, horizon_type, and
        horizon_value columns added.
    """
    n = int(horizon.value)

    if by_cols:
        # Process each partition separately
        future_mids = np.full(trades.height, np.nan)
        trade_idx_col = "__trade_idx__"
        trades_indexed = trades.with_row_index(trade_idx_col)

        for group_keys, trades_group in trades_indexed.group_by(by_cols):
            # Build filter for this partition's quotes
            if len(by_cols) == 1:
                gk = (group_keys,) if not isinstance(group_keys, tuple) else group_keys
            else:
                gk = group_keys if isinstance(group_keys, tuple) else (group_keys,)

            q_filter = pl.lit(True)
            for col, val in zip(by_cols, gk, strict=True):
                q_filter = q_filter & (pl.col(col) == val)
            q_part = quotes.filter(q_filter).sort("timestamp")

            q_ts = q_part["timestamp"].cast(pl.Int64).to_numpy()
            q_mids = q_part["mid"].to_numpy()
            t_ts = trades_group["timestamp"].cast(pl.Int64).to_numpy()
            t_indices = trades_group[trade_idx_col].to_numpy()

            part_mids = _tick_clock_partition(t_ts, q_ts, q_mids, n)
            future_mids[t_indices] = part_mids

        result = trades.with_columns(
            pl.Series("future_mid", future_mids, dtype=pl.Float64, nan_to_null=True),
        )
    else:
        q_ts = quotes["timestamp"].cast(pl.Int64).to_numpy()
        q_mids = quotes["mid"].to_numpy()
        t_ts = trades["timestamp"].cast(pl.Int64).to_numpy()

        future_mids_arr = _tick_clock_partition(t_ts, q_ts, q_mids, n)
        result = trades.with_columns(
            pl.Series("future_mid", future_mids_arr, dtype=pl.Float64, nan_to_null=True),
        )

    null_mid = pl.col("mid").is_null() | (pl.col("mid") == 0)

    if unit == Unit.BPS:
        markout_expr = (
            pl.when(null_mid | pl.col("future_mid").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("side").cast(pl.Float64)
                * (pl.col("future_mid") - pl.col("mid"))
                / pl.col("mid")
                * 10_000
            )
        )
    else:
        markout_expr = (
            pl.when(null_mid | pl.col("future_mid").is_null())
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(
                pl.col("side").cast(pl.Float64) * (pl.col("future_mid") - pl.col("mid"))
            )
        )

    return result.with_columns(
        markout_expr.alias("markout"),
        pl.lit(horizon.type.value).alias("horizon_type"),
        pl.lit(horizon.value).alias("horizon_value"),
    )


def compute(
    trades: pl.DataFrame,
    quotes: pl.DataFrame | None = None,
    *,
    horizons: HorizonSet,
    unit: str = "bps",
    by: str | list[str] | None = None,
    perspective: str = "taker",
) -> MarkoutResult:
    """Compute markout P&L for a set of trades against quote data.

    Args:
        trades: DataFrame of trade records. Must contain timestamp, side,
            price, and mid columns.
        quotes: DataFrame of quote records. Must contain timestamp and mid
            columns. Required for wall and tick horizon types.
        horizons: Set of horizons specifying when to measure markout.
        unit: Output unit for markout values. One of "bps" or "price".
        by: Optional column name or list of column names to group results by.
        perspective: Whose P&L to measure. "taker" (default) means positive
            markout when price moves in the trade direction. "maker" flips
            the sign so positive markout means the liquidity provider profited.

    Returns:
        MarkoutResult with markout data per trade per horizon.

    Raises:
        ValueError: If required columns are missing, incompatible dtypes are
            detected, or configuration is invalid.
    """
    if perspective not in ("taker", "maker"):
        msg = f"perspective must be 'taker' or 'maker', got {perspective!r}"
        raise ValueError(msg)

    trades = maybe_to_polars(trades)
    if quotes is not None:
        quotes = maybe_to_polars(quotes)

    _validate_inputs(trades, quotes, horizons, by)

    parsed_unit = Unit(unit)
    by_cols = [by] if isinstance(by, str) else by

    # Sort quotes once before the loop
    if quotes is not None:
        quotes = quotes.sort("timestamp")

    results: list[pl.DataFrame] = []
    for horizon in horizons:
        if horizon.type == HorizonType.WALL:
            assert quotes is not None  # guaranteed by validation
            chunk = _compute_wall_clock(trades, quotes, horizon, parsed_unit, by_cols)
            results.append(chunk)
        elif horizon.type == HorizonType.TRADE:
            chunk = _compute_trade_clock(trades, horizon, parsed_unit, by_cols)
            results.append(chunk)
        else:
            assert quotes is not None  # guaranteed by validation
            chunk = _compute_tick_clock(trades, quotes, horizon, parsed_unit, by_cols)
            results.append(chunk)

    combined = pl.concat(results) if len(results) > 1 else results[0]

    if perspective == "maker":
        combined = combined.with_columns(
            pl.when(pl.col("markout").is_not_null())
            .then(-pl.col("markout"))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("markout")
        )

    return MarkoutResult(data=combined, unit=parsed_unit.value, perspective=perspective)
