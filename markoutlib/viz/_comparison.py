"""Markout comparison: small-multiples per segment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from markoutlib.viz._style import COLORS, apply_style

if TYPE_CHECKING:
    from markoutlib._result import MarkoutResult

_MAX_COLS = 3


def plot_comparison(result: MarkoutResult, *, by: str) -> go.Figure:
    """Plot markout curves as small multiples, one panel per segment.

    Args:
        result: MarkoutResult instance to pull curve data from.
        by: Column name whose unique values define individual panels.

    Returns:
        Plotly Figure with one subplot per segment, shared y-axis.
    """
    curve_df = result.curve(by=by, n_bootstrap=500)

    segments = curve_df[by].unique(maintain_order=True).to_list()
    n_segs = len(segments)
    n_cols = min(n_segs, _MAX_COLS)
    n_rows = math.ceil(n_segs / n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[str(s) for s in segments],
        shared_yaxes=True,
    )

    for idx, seg in enumerate(segments):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = COLORS[idx % len(COLORS)]

        seg_df = curve_df.filter(pl.col(by) == seg).sort("horizon_value")
        horizons = seg_df["horizon_value"].to_list()
        means = seg_df["markout_mean"].to_list()
        n_obs_total = int(seg_df["n_obs"].sum())

        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=means,
                mode="lines+markers",
                name=str(seg),
                line=dict(color=color),
                marker=dict(color=color, size=5),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # y=0 reference line
        if horizons:
            fig.add_trace(
                go.Scatter(
                    x=[horizons[0], horizons[-1]],
                    y=[0.0, 0.0],
                    mode="lines",
                    line=dict(color="black", dash="dash", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                    name="zero",
                ),
                row=row,
                col=col,
            )

        # n_obs annotation
        axis_suffix = "" if (row == 1 and col == 1) else str(idx + 1)
        fig.add_annotation(
            text=f"n={n_obs_total:,}",
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            x=0.98,
            y=0.98,
            showarrow=False,
            xanchor="right",
            yanchor="top",
            font=dict(size=10),
            row=row,
            col=col,
        )

    apply_style(fig, "Markout Comparison")
    return fig
