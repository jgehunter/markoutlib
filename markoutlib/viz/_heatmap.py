"""Markout heatmap: segment × horizon."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from markoutlib.viz._style import DIVERGING_COLORSCALE, apply_style

if TYPE_CHECKING:
    from markoutlib._result import MarkoutResult


def plot_heatmap(result: MarkoutResult, *, by: str) -> go.Figure:
    """Plot a heatmap of mean markout across segments and horizons.

    Args:
        result: MarkoutResult instance to pull curve data from.
        by: Column name used to define heatmap rows (segments).

    Returns:
        Plotly Figure with a single Heatmap trace.
    """
    curve_df = result.curve(by=by, n_bootstrap=100)

    # Pivot: rows = segment, columns = horizon_value, values = markout_mean
    pivot = (
        curve_df.select([by, "horizon_value", "markout_mean"])
        .pivot(on="horizon_value", index=by, values="markout_mean")
        .sort(by)
    )

    segments = pivot[by].to_list()
    horizon_cols = [c for c in pivot.columns if c != by]
    horizon_labels = [float(c) for c in horizon_cols]

    z = pivot.select(horizon_cols).to_numpy().astype(float)

    abs_max = float(np.nanmax(np.abs(z)))
    zmin = -abs_max
    zmax = abs_max

    fig = go.Figure(
        go.Heatmap(
            x=horizon_labels,
            y=segments,
            z=z,
            colorscale=DIVERGING_COLORSCALE,
            zmid=0.0,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Markout"),
        )
    )

    apply_style(fig, "Markout Heatmap")
    fig.update_layout(
        xaxis=dict(title="Horizon"),
        yaxis=dict(title=by),
    )
    return fig
