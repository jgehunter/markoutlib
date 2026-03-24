"""Core markout computation engine."""

from __future__ import annotations

import polars as pl

from markoutlib._compat import maybe_to_polars
from markoutlib._horizons import HorizonSet
from markoutlib._types import (
    REQUIRED_QUOTE_COLS,
    REQUIRED_TRADE_COLS,
    HorizonType,
)


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


def compute(
    trades: pl.DataFrame,
    quotes: pl.DataFrame | None = None,
    *,
    horizons: HorizonSet,
    unit: str = "bps",
    by: str | list[str] | None = None,
) -> pl.DataFrame:
    """Compute markout P&L for a set of trades against quote data.

    Args:
        trades: DataFrame of trade records. Must contain timestamp, side,
            price, and mid columns.
        quotes: DataFrame of quote records. Must contain timestamp and mid
            columns. Required for wall and tick horizon types.
        horizons: Set of horizons specifying when to measure markout.
        unit: Output unit for markout values. One of "bps" or "price".
        by: Optional column name or list of column names to group results by.

    Returns:
        DataFrame with markout results per trade per horizon.

    Raises:
        ValueError: If required columns are missing, incompatible dtypes are
            detected, or configuration is invalid.
        NotImplementedError: Until the computation engine is implemented.
    """
    trades = maybe_to_polars(trades)
    if quotes is not None:
        quotes = maybe_to_polars(quotes)

    _validate_inputs(trades, quotes, horizons, by)

    raise NotImplementedError
