"""Markout distribution histogram."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

from markoutlib._horizons import Horizon, HorizonSet
from markoutlib.viz._style import COLORS, apply_style

if TYPE_CHECKING:
    from markoutlib._result import MarkoutResult


def plot_distribution(
    result: MarkoutResult,
    *,
    horizon: Horizon | HorizonSet,
    by: str | None = None,
) -> go.Figure:
    """Plot the markout distribution as a histogram for a single horizon.

    Args:
        result: MarkoutResult instance containing per-trade markout data.
        horizon: A single Horizon or a HorizonSet with exactly one element.
        by: Optional column name to overlay per-segment histograms.

    Returns:
        Plotly Figure with overlaid histogram traces.

    Raises:
        ValueError: If horizon is a HorizonSet with != 1 element.
    """
    h = horizon.single() if isinstance(horizon, HorizonSet) else horizon

    data = result.to_polars()
    filtered = data.filter(
        (pl.col("horizon_type") == h.type.value) & (pl.col("horizon_value") == h.value)
    )

    fig = go.Figure()

    if by is None:
        markouts = filtered["markout"].drop_nulls().to_list()
        fig.add_trace(
            go.Histogram(
                x=markouts,
                nbinsx=50,
                name="markout",
                marker_color=COLORS[0],
            )
        )
        barmode = "overlay"
    else:
        segments = filtered[by].unique(maintain_order=True).to_list()
        for idx, seg in enumerate(segments):
            seg_markouts = (
                filtered.filter(pl.col(by) == seg)["markout"].drop_nulls().to_list()
            )
            fig.add_trace(
                go.Histogram(
                    x=seg_markouts,
                    nbinsx=50,
                    name=str(seg),
                    opacity=0.6,
                    marker_color=COLORS[idx % len(COLORS)],
                )
            )
        barmode = "overlay"

    fig.update_layout(barmode=barmode)
    apply_style(fig, f"Markout Distribution (horizon={h.value})")
    fig.update_layout(
        xaxis=dict(title="Markout"),
        yaxis=dict(title="Count"),
    )
    return fig
