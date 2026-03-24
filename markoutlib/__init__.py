"""Markout P&L analysis for Python."""

__version__ = "0.1.0"

from markoutlib._compute import compute
from markoutlib._horizons import Horizon, HorizonSet, seconds, ticks, trades
from markoutlib._result import MarkoutResult

__all__ = [
    "compute",
    "seconds",
    "trades",
    "ticks",
    "Horizon",
    "HorizonSet",
    "MarkoutResult",
    "__version__",
]
