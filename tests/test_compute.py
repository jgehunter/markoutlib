"""Tests for core markout computation."""

from datetime import datetime, timedelta

import polars as pl

from markoutlib._horizons import seconds


def _make_known_answer_data():
    """1000 buys where mid always moves up exactly 1 bps at t+5s.

    Uses mid=1024.0 (power of 2) for clean floating point.
    future_mid = 1024.0 * (1 + 1e-4) = 1024.1024
    markout_bps = 1 * (1024.1024 - 1024.0) / 1024.0 * 10000 = 1.0
    """
    base = datetime(2024, 1, 15, 10, 0, 0)
    n = 1000
    mid = 1024.0
    future_mid = mid * (1 + 1e-4)

    trades = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i * 10) for i in range(n)],
            "side": [1] * n,
            "price": [mid + 0.01] * n,
            "mid": [mid] * n,
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quote_times = []
    quote_mids = []
    for i in range(n):
        t = base + timedelta(seconds=i * 10)
        quote_times.append(t)
        quote_mids.append(mid)
        quote_times.append(t + timedelta(seconds=5))
        quote_mids.append(future_mid)

    quotes = pl.DataFrame(
        {"timestamp": quote_times, "mid": quote_mids}
    ).cast({"timestamp": pl.Datetime("us")}).sort("timestamp")

    return trades, quotes


def test_known_answer_wall_clock():
    from markoutlib._compute import compute

    trades, quotes = _make_known_answer_data()
    result = compute(trades=trades, quotes=quotes, horizons=seconds(5))
    df = result.to_polars()

    assert df.shape[0] == 1000
    markouts = df["markout"].drop_nulls().to_list()
    mean_markout = sum(markouts) / len(markouts)
    assert abs(mean_markout - 1.0) < 1e-6, f"expected ~1.0 bps, got {mean_markout}"


def test_null_propagation_wall_clock():
    from markoutlib._compute import compute

    base = datetime(2024, 1, 15, 10, 0, 0)
    trades = pl.DataFrame(
        {
            "timestamp": [
                base,
                base + timedelta(seconds=100),
            ],
            "side": [1, 1],
            "price": [1024.0, 1024.0],
            "mid": [1024.0, 1024.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    # Only quotes up to t+10s
    quotes = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i) for i in range(11)],
            "mid": [1024.0 + 0.001 * i for i in range(11)],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades, quotes=quotes, horizons=seconds(5))
    df = result.to_polars()

    # First trade at t=0, horizon t+5 -- quote exists
    row0 = df.filter(pl.col("timestamp") == base)
    assert row0["markout"][0] is not None

    # Second trade at t=100, horizon t+105 -- no quote exists
    row1 = df.filter(pl.col("timestamp") == base + timedelta(seconds=100))
    assert row1["markout"][0] is None
    assert row1["future_mid"][0] is None


def test_auditability():
    from markoutlib._compute import compute

    trades, quotes = _make_known_answer_data()
    result = compute(trades=trades, quotes=quotes, horizons=seconds(5))
    df = result.to_polars().filter(pl.col("markout").is_not_null())

    recomputed = (
        df["side"] * (df["future_mid"] - df["mid"]) / df["mid"] * 10_000
    )
    diff = (df["markout"] - recomputed).abs().max()
    assert diff < 1e-10, f"auditability check failed, max diff = {diff}"
