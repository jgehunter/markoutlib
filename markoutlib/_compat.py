"""Pandas <-> Polars conversion layer."""

from __future__ import annotations

from typing import Any

import polars as pl


def _get_pandas():
    """Lazy import pandas. Returns module or None."""
    try:
        import pandas as pd

        return pd
    except ImportError:
        return None


def maybe_to_polars(df: Any) -> pl.DataFrame:
    """Convert input to pl.DataFrame if needed."""
    if isinstance(df, pl.DataFrame):
        return df
    pd = _get_pandas()
    if pd is not None and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    msg = "expected polars or pandas DataFrame"
    raise TypeError(msg)


def to_pandas(df: pl.DataFrame):
    """Convert pl.DataFrame to pandas. Raises ImportError if pandas missing."""
    pd = _get_pandas()
    if pd is None:
        msg = "Install pandas to use .to_pandas(): pip install markoutlib[pandas]"
        raise ImportError(msg)
    return df.to_pandas()
