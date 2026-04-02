"""Tests for core markout computation."""

from datetime import datetime, timedelta

import polars as pl
import pytest

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

    quotes = (
        pl.DataFrame({"timestamp": quote_times, "mid": quote_mids})
        .cast({"timestamp": pl.Datetime("us")})
        .sort("timestamp")
    )

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

    recomputed = df["side"] * (df["future_mid"] - df["mid"]) / df["mid"] * 10_000
    diff = (df["markout"] - recomputed).abs().max()
    assert diff < 1e-10, f"auditability check failed, max diff = {diff}"


def test_trade_clock_basic():
    from markoutlib._compute import compute
    from markoutlib._horizons import trades as trades_h

    base = datetime(2024, 1, 15, 10, 0, 0)
    t = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i) for i in range(10)],
            "side": [1] * 10,
            "price": [100.0] * 10,
            "mid": [100.0 + i for i in range(10)],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=t, quotes=None, horizons=trades_h(3))
    df = result.to_polars()

    # Trade 0: mid=100, future_mid=103 (trade 3's mid)
    row0 = df.row(0, named=True)
    assert row0["future_mid"] == 103.0
    assert row0["markout"] is not None

    # Trade 7: mid=107, only 2 trades forward — null
    row7 = df.filter(pl.col("mid") == 107.0)
    assert row7["future_mid"][0] is None
    assert row7["markout"][0] is None


def test_trade_clock_partitioned():
    from markoutlib._compute import compute
    from markoutlib._horizons import trades as trades_h

    base = datetime(2024, 1, 15, 10, 0, 0)
    rows = []
    for i in range(20):
        sym = "AAPL" if i % 2 == 0 else "MSFT"
        mid = 100.0 + (i // 2) if sym == "AAPL" else 200.0 - (i // 2)
        rows.append(
            {
                "timestamp": base + timedelta(seconds=i),
                "side": 1,
                "price": mid,
                "mid": mid,
                "symbol": sym,
            }
        )

    t = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=t, quotes=None, horizons=trades_h(2), by="symbol")
    df = result.to_polars()

    # AAPL trades should only see AAPL future mids (going up)
    aapl = df.filter(pl.col("symbol") == "AAPL").filter(pl.col("markout").is_not_null())
    assert all(m > 0 for m in aapl["markout"].to_list())

    # MSFT trades should only see MSFT future mids (going down)
    msft = df.filter(pl.col("symbol") == "MSFT").filter(pl.col("markout").is_not_null())
    assert all(m < 0 for m in msft["markout"].to_list())


def test_tick_clock_basic():
    from markoutlib._compute import compute
    from markoutlib._horizons import ticks

    base = datetime(2024, 1, 15, 10, 0, 0)

    trades_df = pl.DataFrame(
        {
            "timestamp": [base],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    # 5 quotes: one AT trade time (should NOT count), four AFTER
    quotes_df = pl.DataFrame(
        {
            "timestamp": [
                base,
                base + timedelta(milliseconds=100),
                base + timedelta(milliseconds=200),
                base + timedelta(milliseconds=300),
                base + timedelta(milliseconds=400),
            ],
            "mid": [100.0, 100.5, 101.0, 101.5, 102.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes_df, horizons=ticks(2))
    df = result.to_polars()

    # ticks(2) = 2nd quote strictly after trade = +200ms = mid 101.0
    assert df["future_mid"][0] == 101.0


def test_tick_clock_null_when_insufficient_ticks():
    from markoutlib._compute import compute
    from markoutlib._horizons import ticks

    base = datetime(2024, 1, 15, 10, 0, 0)

    trades_df = pl.DataFrame(
        {
            "timestamp": [base],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes_df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(milliseconds=100)],
            "mid": [100.5],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes_df, horizons=ticks(5))
    df = result.to_polars()

    assert df["future_mid"][0] is None
    assert df["markout"][0] is None


def test_tick_clock_partitioned():
    from markoutlib._compute import compute
    from markoutlib._horizons import ticks

    base = datetime(2024, 1, 15, 10, 0, 0)

    trades_df = pl.DataFrame(
        {
            "timestamp": [base, base],
            "side": [1, 1],
            "price": [100.0, 200.0],
            "mid": [100.0, 200.0],
            "symbol": ["AAPL", "MSFT"],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes_df = pl.DataFrame(
        {
            "timestamp": [
                base + timedelta(milliseconds=100),
                base + timedelta(milliseconds=100),
                base + timedelta(milliseconds=200),
                base + timedelta(milliseconds=200),
            ],
            "mid": [101.0, 199.0, 102.0, 198.0],
            "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes_df, horizons=ticks(1), by="symbol")
    df = result.to_polars()

    aapl = df.filter(pl.col("symbol") == "AAPL")
    msft = df.filter(pl.col("symbol") == "MSFT")

    assert aapl["future_mid"][0] == 101.0
    assert msft["future_mid"][0] == 199.0


def test_trade_clock_zero():
    from markoutlib._compute import compute
    from markoutlib._horizons import trades as trades_h

    base = datetime(2024, 1, 15, 10, 0, 0)
    t = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i) for i in range(5)],
            "side": [1] * 5,
            "price": [100.0] * 5,
            "mid": [100.0 + i for i in range(5)],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=t, quotes=None, horizons=trades_h(0))
    df = result.to_polars()

    for i in range(5):
        assert df["future_mid"][i] == df["mid"][i]
        assert abs(df["markout"][i]) < 1e-10


def test_trade_clock_negative():
    from markoutlib._compute import compute
    from markoutlib._horizons import trades as trades_h

    base = datetime(2024, 1, 15, 10, 0, 0)
    t = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i) for i in range(10)],
            "side": [1] * 10,
            "price": [100.0] * 10,
            "mid": [100.0 + i for i in range(10)],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=t, quotes=None, horizons=trades_h(-3))
    df = result.to_polars()

    # Trade 5: mid=105, 3 trades back is trade 2 with mid=102
    assert df["future_mid"][5] == 102.0

    # First 3 trades: not enough prior trades
    for i in range(3):
        assert df["future_mid"][i] is None
        assert df["markout"][i] is None


def test_wall_clock_zero():
    from markoutlib._compute import compute

    trades_df, quotes = _make_known_answer_data()
    result = compute(trades=trades_df, quotes=quotes, horizons=seconds(0))
    df = result.to_polars()

    non_null = df.filter(pl.col("markout").is_not_null())
    assert non_null.height > 0
    # Markout at horizon 0 should be ~0
    assert abs(non_null["markout"].mean()) < 0.01


def test_wall_clock_negative():
    from markoutlib._compute import compute

    base = datetime(2024, 1, 15, 10, 0, 0)
    trades_df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=30)],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=i) for i in range(61)],
            "mid": [99.0 + 0.001 * i for i in range(61)],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    # seconds(-10): target = t+30 - 10 = t+20, quote mid at t+20 = 99.020
    result = compute(trades=trades_df, quotes=quotes, horizons=seconds(-10))
    df = result.to_polars()

    assert df["future_mid"][0] is not None
    assert abs(df["future_mid"][0] - 99.020) < 0.01


def test_wall_clock_negative_null_when_no_earlier_quotes():
    from markoutlib._compute import compute

    base = datetime(2024, 1, 15, 10, 0, 0)
    trades_df = pl.DataFrame(
        {
            "timestamp": [base],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=5)],
            "mid": [100.5],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes, horizons=seconds(-10))
    df = result.to_polars()

    assert df["future_mid"][0] is None
    assert df["markout"][0] is None


def test_tick_clock_zero():
    from markoutlib._compute import compute
    from markoutlib._horizons import ticks

    base = datetime(2024, 1, 15, 10, 0, 0)
    trades_df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(milliseconds=150)],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes_df = pl.DataFrame(
        {
            "timestamp": [
                base,
                base + timedelta(milliseconds=100),
                base + timedelta(milliseconds=200),
            ],
            "mid": [99.0, 99.5, 100.5],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes_df, horizons=ticks(0))
    df = result.to_polars()

    # Last quote at or before t+150ms is at t+100ms (mid=99.5)
    assert df["future_mid"][0] == 99.5


def test_tick_clock_zero_no_prior_quotes():
    from markoutlib._compute import compute
    from markoutlib._horizons import ticks

    base = datetime(2024, 1, 15, 10, 0, 0)
    trades_df = pl.DataFrame(
        {
            "timestamp": [base],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes_df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(seconds=1)],
            "mid": [100.5],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes_df, horizons=ticks(0))
    df = result.to_polars()
    assert df["future_mid"][0] is None


def test_tick_clock_negative():
    from markoutlib._compute import compute
    from markoutlib._horizons import ticks

    base = datetime(2024, 1, 15, 10, 0, 0)
    trades_df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(milliseconds=500)],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes_df = pl.DataFrame(
        {
            "timestamp": [
                base,
                base + timedelta(milliseconds=100),
                base + timedelta(milliseconds=200),
                base + timedelta(milliseconds=300),
                base + timedelta(milliseconds=400),
                base + timedelta(milliseconds=600),
            ],
            "mid": [90.0, 91.0, 92.0, 93.0, 94.0, 95.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes_df, horizons=ticks(-2))
    df = result.to_polars()

    # bisect_right gives idx=5, last_before=4, target=4+(-2)=2, mid=92.0
    assert df["future_mid"][0] == 92.0


def test_tick_clock_negative_insufficient():
    from markoutlib._compute import compute
    from markoutlib._horizons import ticks

    base = datetime(2024, 1, 15, 10, 0, 0)
    trades_df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(milliseconds=150)],
            "side": [1],
            "price": [100.0],
            "mid": [100.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    quotes_df = pl.DataFrame(
        {
            "timestamp": [base + timedelta(milliseconds=100)],
            "mid": [99.5],
        }
    ).cast({"timestamp": pl.Datetime("us")})

    result = compute(trades=trades_df, quotes=quotes_df, horizons=ticks(-5))
    df = result.to_polars()
    assert df["future_mid"][0] is None


def test_perspective_maker_negates_markout():
    from markoutlib._compute import compute

    trades, quotes = _make_known_answer_data()
    taker = compute(trades=trades, quotes=quotes, horizons=seconds(5))
    maker = compute(
        trades=trades, quotes=quotes, horizons=seconds(5), perspective="maker"
    )

    taker_df = taker.to_polars()
    maker_df = maker.to_polars()

    # All markouts should be negated
    t_marks = taker_df["markout"].drop_nulls().to_list()
    m_marks = maker_df["markout"].drop_nulls().to_list()
    assert len(t_marks) == len(m_marks)
    for t, m in zip(t_marks, m_marks, strict=True):
        assert abs(t + m) < 1e-10, f"expected negation: taker={t}, maker={m}"


def test_perspective_maker_preserves_future_mid():
    from markoutlib._compute import compute

    trades, quotes = _make_known_answer_data()
    taker = compute(trades=trades, quotes=quotes, horizons=seconds(5))
    maker = compute(
        trades=trades, quotes=quotes, horizons=seconds(5), perspective="maker"
    )

    # future_mid should be identical — only markout sign changes
    assert (
        taker.to_polars()["future_mid"].to_list()
        == maker.to_polars()["future_mid"].to_list()
    )


def test_perspective_invalid_raises():
    from markoutlib._compute import compute

    trades, quotes = _make_known_answer_data()
    with pytest.raises(ValueError, match="perspective"):
        compute(
            trades=trades, quotes=quotes, horizons=seconds(5), perspective="neutral"
        )


def test_tick_clock_native_matches_numpy():
    """When the Rust extension is available, verify it matches numpy."""
    import numpy as np

    from markoutlib._compute import (
        _USE_NATIVE,
        _tick_clock_partition_np,
    )

    if not _USE_NATIVE:
        pytest.skip("Rust extension not installed")

    from _markoutlib_native import tick_clock_partition as rs_fn

    rng = np.random.default_rng(42)
    n_trades = 10_000
    n_quotes = 50_000

    trade_ts = np.sort(rng.integers(0, 1_000_000, size=n_trades)).astype(np.int64)
    quote_ts = np.sort(rng.integers(0, 1_000_000, size=n_quotes)).astype(np.int64)
    quote_mids = rng.uniform(99.0, 101.0, size=n_quotes)

    for n in [-5, -1, 0, 1, 5, 50]:
        np_result = _tick_clock_partition_np(trade_ts, quote_ts, quote_mids, n)
        rs_result = np.asarray(rs_fn(trade_ts, quote_ts, quote_mids, n))

        # Both should be NaN in the same positions
        np_nan = np.isnan(np_result)
        rs_nan = np.isnan(rs_result)
        assert np.array_equal(np_nan, rs_nan), f"NaN mismatch at n={n}"

        # Non-NaN values should be identical
        valid = ~np_nan
        assert np.allclose(
            np_result[valid],
            rs_result[valid],
        ), f"Value mismatch at n={n}"
