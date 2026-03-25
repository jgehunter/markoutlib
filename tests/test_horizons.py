"""Tests for horizon constructors and HorizonSet composition."""

import pytest

from markoutlib._horizons import (
    Horizon,
    HorizonSet,
    seconds,
    seconds_range,
    ticks,
    ticks_range,
    trades,
    trades_range,
)
from markoutlib._types import HorizonType


def test_seconds_returns_horizon_set():
    hs = seconds(1, 5, 30)
    assert isinstance(hs, HorizonSet)
    assert len(hs) == 3
    assert all(h.type == HorizonType.WALL for h in hs)
    assert [h.value for h in hs] == [1, 5, 30]


def test_trades_returns_horizon_set():
    hs = trades(10, 50)
    assert isinstance(hs, HorizonSet)
    assert len(hs) == 2
    assert all(h.type == HorizonType.TRADE for h in hs)
    assert [h.value for h in hs] == [10, 50]


def test_ticks_returns_horizon_set():
    hs = ticks(100, 500)
    assert isinstance(hs, HorizonSet)
    assert all(h.type == HorizonType.TICK for h in hs)


def test_horizon_set_addition():
    combined = seconds(1, 5) + trades(10) + ticks(100)
    assert len(combined) == 4
    types = [h.type for h in combined]
    assert types == [
        HorizonType.WALL,
        HorizonType.WALL,
        HorizonType.TRADE,
        HorizonType.TICK,
    ]


def test_zero_horizon_allowed():
    hs = seconds(0)
    assert len(hs) == 1
    assert hs[0].value == 0


def test_negative_horizon_allowed():
    hs = seconds(-5, 0, 5)
    assert len(hs) == 3
    assert [h.value for h in hs] == [-5, 0, 5]


def test_negative_trades_allowed():
    hs = trades(-3)
    assert hs[0].value == -3


def test_negative_ticks_allowed():
    hs = ticks(-10)
    assert hs[0].value == -10


def test_single_horizon_from_set():
    hs = seconds(5)
    assert len(hs) == 1
    h = hs.single()
    assert isinstance(h, Horizon)
    assert h.value == 5


def test_single_rejects_multi():
    with pytest.raises(ValueError, match="expected a single horizon"):
        seconds(1, 5).single()


def test_seconds_range_basic():
    hs = seconds_range(1, 10, 3)
    assert [h.value for h in hs] == [1, 4, 7, 10]


def test_seconds_range_inclusive_stop():
    hs = seconds_range(0, 60, 5)
    values = [h.value for h in hs]
    assert values[0] == 0
    assert values[-1] == 60


def test_seconds_range_stop_not_reachable():
    hs = seconds_range(1, 11, 3)
    assert [h.value for h in hs] == [1, 4, 7, 10]


def test_seconds_range_crossing_zero():
    hs = seconds_range(-10, 10, 5)
    assert [h.value for h in hs] == [-10, -5, 0, 5, 10]


def test_trades_range():
    hs = trades_range(5, 25, 5)
    assert [h.value for h in hs] == [5, 10, 15, 20, 25]


def test_ticks_range():
    hs = ticks_range(10, 50, 10)
    assert [h.value for h in hs] == [10, 20, 30, 40, 50]


def test_range_step_must_be_positive():
    with pytest.raises(ValueError, match="step must be positive"):
        seconds_range(1, 10, -1)


def test_range_start_must_be_lte_stop():
    with pytest.raises(ValueError, match="start must be <= stop"):
        seconds_range(10, 1, 1)


def test_range_composable():
    combined = seconds_range(-5, 5, 5) + trades(10)
    assert len(combined) == 4
