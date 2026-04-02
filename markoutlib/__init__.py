"""Markout P&L analysis for Python."""

__version__ = "0.3.0"

from markoutlib._compute import compute
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
from markoutlib._result import MarkoutResult

__all__ = [
    "compute",
    "seconds",
    "seconds_range",
    "trades",
    "trades_range",
    "ticks",
    "ticks_range",
    "Horizon",
    "HorizonSet",
    "MarkoutResult",
    "__version__",
]
