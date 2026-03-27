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

    # Build a unique horizon label that distinguishes types (e.g. "10s" vs "10t")
    _suffix = {"wall": "s", "trade": "t", "tick": "q"}
    curve_df = curve_df.with_columns(
        (
            curve_df["horizon_value"].cast(str)
            + curve_df["horizon_type"].replace_strict(_suffix, default="")
        ).alias("_hlabel")
    )

    # Pivot: rows = segment, columns = horizon label, values = markout_mean
    pivot = (
        curve_df.select([by, "_hlabel", "markout_mean"])
        .pivot(on="_hlabel", index=by, values="markout_mean")
        .sort(by)
    )

    segments = pivot[by].to_list()
    # Preserve the original horizon ordering from curve_df
    ordered_labels = curve_df.select("_hlabel").unique(maintain_order=True).to_series().to_list()
    horizon_cols = [c for c in ordered_labels if c in pivot.columns]
    horizon_labels = horizon_cols

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
