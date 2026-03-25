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
