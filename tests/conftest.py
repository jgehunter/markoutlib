"""Shared test fixtures."""

from datetime import datetime, timedelta

import polars as pl
import pytest


@pytest.fixture()
def simple_trades() -> pl.DataFrame:
    base = datetime(2024, 1, 15, 10, 0, 0)
    n = 100
    return pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i) for i in range(n)],
            "side": [1 if i % 2 == 0 else -1 for i in range(n)],
            "price": [1024.0 + (0.01 if i % 2 == 0 else -0.01) for i in range(n)],
            "mid": [1024.0] * n,
            "size": [100.0] * n,
            "counterparty": [f"cp_{i % 3}" for i in range(n)],
        }
    ).cast({"timestamp": pl.Datetime("us")})


@pytest.fixture()
def simple_quotes() -> pl.DataFrame:
    base = datetime(2024, 1, 15, 10, 0, 0)
    n = 200
    return pl.DataFrame(
        {
            "timestamp": [
                base + timedelta(milliseconds=500 * i) for i in range(n)
            ],
            "mid": [1024.0 + 0.001 * i for i in range(n)],
        }
    ).cast({"timestamp": pl.Datetime("us")})
