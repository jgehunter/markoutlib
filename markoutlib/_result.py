"""MarkoutResult -- the primary return type of compute()."""

from __future__ import annotations

import polars as pl


class MarkoutResult:
    """Holds per-trade markout data."""

    def __init__(self, data: pl.DataFrame, unit: str) -> None:
        self._data = data
        self._unit = unit

    def to_polars(self) -> pl.DataFrame:
        return self._data

    def to_pandas(self):
        from markoutlib._compat import to_pandas

        return to_pandas(self._data)
