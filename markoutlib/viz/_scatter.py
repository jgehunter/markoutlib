"""Markout scatter plot: arbitrary column vs markout at a horizon."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

from markoutlib._horizons import Horizon, HorizonSet
from markoutlib.viz._style import COLORS, apply_style

if TYPE_CHECKING:
    from markoutlib._result import MarkoutResult


def plot_scatter(
    result: MarkoutResult,
    *,
    x: str,
    horizon: Horizon | HorizonSet,
) -> go.Figure:
    """Plot an arbitrary column against markout for a single horizon.

    Args:
        result: MarkoutResult instance containing per-trade markout data.
        x: Column name to plot on the x-axis.
        horizon: A single Horizon or a HorizonSet with exactly one element.

    Returns:
        Plotly Figure with a Scattergl trace and a y=0 reference line.

    Raises:
        ValueError: If horizon is a HorizonSet with != 1 element.
    """
    h = horizon.single() if isinstance(horizon, HorizonSet) else horizon

    data = result.to_polars()
    filtered = data.filter(
        (pl.col("horizon_type") == h.type.value) & (pl.col("horizon_value") == h.value)
    ).drop_nulls(subset=[x, "markout"])

    x_vals = filtered[x].to_list()
    y_vals = filtered["markout"].to_list()

    x_min = min(x_vals) if x_vals else 0.0
    x_max = max(x_vals) if x_vals else 1.0

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=dict(color=COLORS[0], size=3, opacity=0.4),
            name="markout",
        )
    )

    # y=0 reference line
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[0.0, 0.0],
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
            name="zero",
        )
    )

    apply_style(fig, f"Markout vs {x} (horizon={h.value})")
    fig.update_layout(
        xaxis=dict(title=x),
        yaxis=dict(title="Markout"),
    )
    return fig
