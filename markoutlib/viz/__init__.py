"""Visualization module for markoutlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markoutlib._result import MarkoutResult


class PlotAccessor:
    """Namespace for plot methods, accessed via result.plot."""

    def __init__(self, result: MarkoutResult) -> None:
        self._result = result

    def curve(self, *, by: str | None = None):
        from markoutlib.viz._curve import plot_curve

        return plot_curve(self._result, by=by)

    def heatmap(self, *, by: str):
        from markoutlib.viz._heatmap import plot_heatmap

        return plot_heatmap(self._result, by=by)

    def distribution(self, *, horizon, by: str | None = None):
        from markoutlib.viz._distribution import plot_distribution

        return plot_distribution(self._result, horizon=horizon, by=by)

    def comparison(self, *, by: str):
        from markoutlib.viz._comparison import plot_comparison

        return plot_comparison(self._result, by=by)

    def scatter(self, *, x: str, horizon):
        from markoutlib.viz._scatter import plot_scatter

        return plot_scatter(self._result, x=x, horizon=horizon)
