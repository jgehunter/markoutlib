"""Tests for horizon constructors and HorizonSet composition."""

import pytest

from markoutlib._horizons import Horizon, HorizonSet, seconds, ticks, trades
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


def test_zero_horizon_rejected():
    with pytest.raises(ValueError, match="horizon values must be positive"):
        seconds(0)


def test_negative_horizon_rejected():
    with pytest.raises(ValueError, match="horizon values must be positive"):
        trades(-5)


def test_single_horizon_from_set():
    hs = seconds(5)
    assert len(hs) == 1
    h = hs.single()
    assert isinstance(h, Horizon)
    assert h.value == 5


def test_single_rejects_multi():
    with pytest.raises(ValueError, match="expected a single horizon"):
        seconds(1, 5).single()
