"""Horizon types and constructors."""

from __future__ import annotations

from dataclasses import dataclass

from markoutlib._types import HorizonType


@dataclass(frozen=True, slots=True)
class Horizon:
    """A single markout horizon."""

    type: HorizonType
    value: int | float


class HorizonSet:
    """An ordered collection of horizons. Composable via +."""

    __slots__ = ("_horizons",)

    def __init__(self, horizons: list[Horizon]) -> None:
        self._horizons = horizons

    def __add__(self, other: HorizonSet) -> HorizonSet:
        if not isinstance(other, HorizonSet):
            return NotImplemented
        return HorizonSet(self._horizons + other._horizons)

    def __iter__(self):
        return iter(self._horizons)

    def __len__(self) -> int:
        return len(self._horizons)

    def __getitem__(self, idx: int) -> Horizon:
        return self._horizons[idx]

    def __repr__(self) -> str:
        return f"HorizonSet({self._horizons!r})"

    def single(self) -> Horizon:
        """Return the sole horizon, or raise if this set has != 1 element."""
        if len(self._horizons) != 1:
            msg = f"expected a single horizon, got {len(self._horizons)}"
            raise ValueError(msg)
        return self._horizons[0]


def _make_horizons(type_: HorizonType, *values: int | float) -> HorizonSet:
    for v in values:
        if v <= 0:
            msg = "horizon values must be positive"
            raise ValueError(msg)
    return HorizonSet([Horizon(type=type_, value=v) for v in values])


def seconds(*values: int | float) -> HorizonSet:
    """Wall-clock horizons in seconds."""
    return _make_horizons(HorizonType.WALL, *values)


def trades(*values: int | float) -> HorizonSet:
    """Trade-clock horizons (N trades forward)."""
    return _make_horizons(HorizonType.TRADE, *values)


def ticks(*values: int | float) -> HorizonSet:
    """Tick-clock horizons (N quote updates forward)."""
    return _make_horizons(HorizonType.TICK, *values)
