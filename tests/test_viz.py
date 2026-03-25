"""Tests for visualization methods."""

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import polars as pl
import pytest

from markoutlib._result import MarkoutResult


@pytest.fixture()
def viz_result():
    rng = np.random.default_rng(42)
    rows = []
    for h in [1.0, 5.0, 30.0]:
        for i in range(100):
            rows.append(
                {
                    "timestamp": datetime(2024, 1, 15, 10, 0, 0) + timedelta(seconds=i),
                    "side": 1,
                    "price": 100.0,
                    "mid": 100.0,
                    "size": float(rng.uniform(10, 1000)),
                    "counterparty": f"cp_{i % 3}",
                    "horizon_type": "wall",
                    "horizon_value": h,
                    "future_mid": 100.0 + rng.normal(0.01, 0.005),
                    "markout": float(rng.normal(1.0 * (1 - np.exp(-h / 10)), 0.3)),
                }
            )
    data = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us")})
    return MarkoutResult(data, "bps")


def test_curve_returns_figure(viz_result):
    fig = viz_result.plot.curve()
    assert isinstance(fig, go.Figure)


def test_curve_with_by(viz_result):
    fig = viz_result.plot.curve(by="counterparty")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3


def test_heatmap_returns_figure(viz_result):
    fig = viz_result.plot.heatmap(by="counterparty")
    assert isinstance(fig, go.Figure)


def test_distribution_returns_figure(viz_result):
    from markoutlib._horizons import seconds

    fig = viz_result.plot.distribution(horizon=seconds(5))
    assert isinstance(fig, go.Figure)


def test_comparison_returns_figure(viz_result):
    fig = viz_result.plot.comparison(by="counterparty")
    assert isinstance(fig, go.Figure)


def test_scatter_returns_figure(viz_result):
    from markoutlib._horizons import seconds

    fig = viz_result.plot.scatter(x="size", horizon=seconds(5))
    assert isinstance(fig, go.Figure)


def test_curve_linear_axis_with_negative_horizons():
    rng = np.random.default_rng(42)
    rows = []
    for h in [-5.0, 0.0, 5.0]:
        for i in range(50):
            rows.append({
                "timestamp": datetime(2024, 1, 15, 10, 0, 0) + timedelta(seconds=i),
                "side": 1, "price": 100.0, "mid": 100.0, "size": 100.0,
                "counterparty": "cp_0",
                "horizon_type": "wall", "horizon_value": h,
                "future_mid": 100.0 + rng.normal(0.01, 0.005),
                "markout": float(rng.normal(0.5, 0.2)),
            })
    data = pl.DataFrame(rows).cast({"timestamp": pl.Datetime("us")})
    result = MarkoutResult(data, "bps")

    fig = result.plot.curve()
    assert isinstance(fig, go.Figure)
    # x-axis should NOT be log type when negative horizons present
    assert fig.layout.xaxis.type != "log"
