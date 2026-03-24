"""Tests for compute() input validation."""

import polars as pl
import pytest

from markoutlib._horizons import seconds, ticks, trades


def test_missing_trades_column(simple_quotes):
    from markoutlib._compute import compute

    bad = pl.DataFrame({"timestamp": [1], "side": [1], "price": [1.0]}).cast(
        {"timestamp": pl.Datetime("us")}
    )
    with pytest.raises(
        ValueError, match="trades DataFrame missing required column: mid"
    ):
        compute(trades=bad, quotes=simple_quotes, horizons=seconds(5))


def test_missing_quotes_column(simple_trades):
    from markoutlib._compute import compute

    bad = pl.DataFrame({"timestamp": [1]}).cast({"timestamp": pl.Datetime("us")})
    with pytest.raises(
        ValueError, match="quotes DataFrame missing required column: mid"
    ):
        compute(trades=simple_trades, quotes=bad, horizons=seconds(5))


def test_quotes_none_with_wall_clock(simple_trades):
    from markoutlib._compute import compute

    with pytest.raises(ValueError, match="quotes DataFrame required for wall horizons"):
        compute(trades=simple_trades, quotes=None, horizons=seconds(5))


def test_quotes_none_with_tick_clock(simple_trades):
    from markoutlib._compute import compute

    with pytest.raises(ValueError, match="quotes DataFrame required for tick horizons"):
        compute(trades=simple_trades, quotes=None, horizons=ticks(10))


def test_quotes_none_ok_with_trade_clock(simple_trades):
    from markoutlib._compute import compute

    # Trade-clock should work without quotes — no error
    result = compute(trades=simple_trades, quotes=None, horizons=trades(5))
    assert result.to_polars().shape[0] == len(simple_trades)


def test_by_column_missing_in_trades(simple_trades, simple_quotes):
    from markoutlib._compute import compute

    with pytest.raises(ValueError, match="by column 'symbol' not found in trades"):
        compute(
            trades=simple_trades, quotes=simple_quotes, horizons=seconds(5), by="symbol"
        )


def test_by_column_missing_in_quotes(simple_trades, simple_quotes):
    from markoutlib._compute import compute

    with pytest.raises(
        ValueError, match="by column 'counterparty' not found in quotes"
    ):
        compute(
            trades=simple_trades,
            quotes=simple_quotes,
            horizons=seconds(5),
            by="counterparty",
        )


def test_timestamp_dtype_mismatch(simple_trades):
    from markoutlib._compute import compute

    quotes_ns = pl.DataFrame({"timestamp": [1000000000], "mid": [1024.0]}).cast(
        {"timestamp": pl.Datetime("ns")}
    )
    with pytest.raises(ValueError, match="timestamp column type mismatch"):
        compute(trades=simple_trades, quotes=quotes_ns, horizons=seconds(5))


def test_side_invalid_values(simple_quotes):
    from markoutlib._compute import compute

    bad = pl.DataFrame(
        {
            "timestamp": [1],
            "side": [2],
            "price": [1024.0],
            "mid": [1024.0],
        }
    ).cast({"timestamp": pl.Datetime("us")})
    with pytest.raises(ValueError, match="side column must contain only 1 and -1"):
        compute(trades=bad, quotes=simple_quotes, horizons=seconds(5))


def test_side_null_rejected(simple_quotes):
    from markoutlib._compute import compute

    bad = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "side": [1, None],
            "price": [1024.0, 1024.0],
            "mid": [1024.0, 1024.0],
        }
    ).cast({"timestamp": pl.Datetime("us"), "side": pl.Int64})
    with pytest.raises(ValueError, match="side column must contain only 1 and -1"):
        compute(trades=bad, quotes=simple_quotes, horizons=seconds(5))
