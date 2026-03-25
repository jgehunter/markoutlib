"""Tests for MarkoutResult methods."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from markoutlib._result import MarkoutResult


@pytest.fixture()
def simple_result():
    n = 200
    rng = np.random.default_rng(42)
    data = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15, 10, 0, 0) + timedelta(seconds=i) for i in range(n)
            ],
            "side": [1] * n,
            "price": [100.0] * n,
            "mid": [100.0] * n,
            "size": rng.uniform(10, 1000, n).tolist(),
            "counterparty": [f"cp_{i % 3}" for i in range(n)],
            "horizon_type": ["wall"] * n,
            "horizon_value": [5.0] * n,
            "future_mid": (100.0 + rng.normal(0.05, 0.02, n)).tolist(),
            "markout": rng.normal(1.0, 0.5, n).tolist(),
        }
    ).cast({"timestamp": pl.Datetime("us")})
    return MarkoutResult(data, "bps")


def test_curve_returns_expected_columns(simple_result):
    df = simple_result.curve()
    expected = {
        "horizon_type",
        "horizon_value",
        "markout_mean",
        "markout_median",
        "markout_ci_lower",
        "markout_ci_upper",
        "markout_q25",
        "markout_q75",
        "skew",
        "kurtosis",
        "t_stat",
        "p_value",
        "n_obs",
    }
    assert expected.issubset(set(df.columns))


def test_curve_by_column(simple_result):
    df = simple_result.curve(by="counterparty")
    assert "counterparty" in df.columns
    assert df.shape[0] == 3


def test_curve_weighted(simple_result):
    unwt = simple_result.curve()
    wt = simple_result.curve(weight="size")
    assert unwt["markout_mean"][0] != wt["markout_mean"][0]


def test_curve_weight_column_missing(simple_result):
    with pytest.raises(ValueError, match="weight column 'notional' not found"):
        simple_result.curve(weight="notional")


def test_half_life_on_exponential_data():
    tau = 10.0
    terminal = 2.0
    rows = []
    for h in [1, 2, 5, 10, 20, 50]:
        for i in range(100):
            rows.append(
                {
                    "timestamp": datetime(2024, 1, 15, 10, 0, 0) + timedelta(seconds=i),
                    "side": 1,
                    "price": 100.0,
                    "mid": 100.0,
                    "horizon_type": "wall",
                    "horizon_value": float(h),
                    "future_mid": (100.0 + terminal * (1 - np.exp(-h / tau)) * 0.01),
                    "markout": terminal * (1 - np.exp(-h / tau)),
                }
            )
    data = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us")})
    result = MarkoutResult(data, "bps")
    hl = result.half_life()
    assert hl.converged
    assert hl.half_life == pytest.approx(tau * np.log(2), rel=0.1)


def test_half_life_flat_does_not_converge(simple_result):
    data = simple_result._data.with_columns(pl.lit(0.0).alias("markout"))
    flat = MarkoutResult(data, "bps")
    hl = flat.half_life()
    assert not hl.converged


def test_half_life_mixed_horizon_types_does_not_converge():
    """Cannot fit decay across wall + trade horizons (different units)."""
    rows = []
    for h_type, h_val in [("wall", 1.0), ("wall", 5.0), ("trade", 10.0)]:
        for i in range(50):
            rows.append(
                {
                    "timestamp": datetime(2024, 1, 15, 10, 0, 0) + timedelta(seconds=i),
                    "side": 1,
                    "price": 100.0,
                    "mid": 100.0,
                    "horizon_type": h_type,
                    "horizon_value": h_val,
                    "future_mid": 100.01,
                    "markout": 1.0,
                }
            )
    data = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us")})
    result = MarkoutResult(data, "bps")
    hl = result.half_life()
    assert not hl.converged


def test_test_returns_expected_columns(simple_result):
    df = simple_result.test("counterparty")
    expected = {
        "segment",
        "segment_n_obs",
        "segment_mean",
        "complement_mean",
        "test_stat",
        "test_p_value",
    }
    assert expected.issubset(set(df.columns))
    assert df.shape[0] == 3


def test_compare_returns_expected_columns(simple_result):
    df = simple_result.compare(weight="size")
    expected = {
        "horizon_type",
        "horizon_value",
        "markout_unweighted",
        "markout_weighted",
        "n_obs",
    }
    assert expected.issubset(set(df.columns))


def test_compare_weight_missing(simple_result):
    with pytest.raises(ValueError, match="weight column"):
        simple_result.compare(weight="notional")


def test_to_polars(simple_result):
    df = simple_result.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert "markout" in df.columns
    assert "future_mid" in df.columns


def test_to_pandas(simple_result):
    pd = pytest.importorskip("pandas")
    pdf = simple_result.to_pandas()
    assert isinstance(pdf, pd.DataFrame)


def test_effective_spread(simple_result):
    df = simple_result.effective_spread()
    assert "effective_spread_mean" in df.columns
    assert "effective_spread_median" in df.columns
    assert "n_obs" in df.columns
    assert df.height == 1


def test_effective_spread_by(simple_result):
    df = simple_result.effective_spread(by="counterparty")
    assert "counterparty" in df.columns
    assert df.height == 3


def test_realized_spread(simple_result):
    from markoutlib._horizons import seconds

    df = simple_result.realized_spread(horizon=seconds(5))
    assert "realized_spread_mean" in df.columns
    assert "horizon_type" in df.columns


def test_price_impact(simple_result):
    from markoutlib._horizons import seconds

    df = simple_result.price_impact(horizon=seconds(5))
    assert "price_impact_mean" in df.columns


def test_spread_decomposition_identity(simple_result):
    from markoutlib._horizons import seconds

    df = simple_result.spread_decomposition(horizon=seconds(5))
    assert "effective_spread_mean" in df.columns
    assert "realized_spread_mean" in df.columns
    assert "price_impact_mean" in df.columns
    for row in df.iter_rows(named=True):
        eff = row["effective_spread_mean"]
        real = row["realized_spread_mean"]
        imp = row["price_impact_mean"]
        assert abs(eff - (real + imp)) < 1e-10


def test_spread_decomposition_horizon_not_found(simple_result):
    from markoutlib._horizons import seconds

    with pytest.raises(ValueError, match="horizon.*not found"):
        simple_result.spread_decomposition(horizon=seconds(999))


def test_half_life_filters_negative_horizons():
    import warnings

    rows = []
    for h in [-10.0, -5.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for i in range(50):
            m = 2.0 * (1 - np.exp(-h / 10.0)) if h > 0 else h * 0.01
            rows.append(
                {
                    "timestamp": datetime(2024, 1, 15, 10, 0, 0) + timedelta(seconds=i),
                    "side": 1,
                    "price": 100.0,
                    "mid": 100.0,
                    "horizon_type": "wall",
                    "horizon_value": h,
                    "future_mid": 100.0,
                    "markout": m,
                }
            )

    data = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us")})
    result = MarkoutResult(data, "bps")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hl = result.half_life()
        assert any("Negative horizons excluded" in str(x.message) for x in w)

    assert hl.converged
