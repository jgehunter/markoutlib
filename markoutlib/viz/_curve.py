"""Markout decay curve plot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from markoutlib.viz._style import COLORS, apply_style

if TYPE_CHECKING:
    from markoutlib._result import MarkoutResult

_LOG_SCALE_TYPES = {"wall"}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color string to an rgba() string.

    Args:
        hex_color: Six-digit hex color, e.g. '#4C72B0'.
        alpha: Opacity value between 0 and 1.

    Returns:
        CSS rgba string suitable for Plotly fill colors.
    """
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _add_traces(
    fig: go.Figure,
    curve_df: pl.DataFrame,
    by: str | None,
    row: int,
    col: int,
) -> None:
    """Add mean line + CI band traces to a subplot cell.

    Args:
        fig: Plotly Figure (created with make_subplots) to add traces to.
        curve_df: Curve DataFrame filtered to one horizon_type.
        by: Column used for segmentation, or None for a single series.
        row: Subplot row (1-indexed).
        col: Subplot column (1-indexed).
    """
    if by is None:
        _add_single_trace(
            fig, curve_df, label="mean", color=COLORS[0], row=row, col=col
        )
    else:
        segments = curve_df[by].unique(maintain_order=True).to_list()
        for idx, seg in enumerate(segments):
            seg_df = curve_df.filter(pl.col(by) == seg).sort("horizon_value")
            color = COLORS[idx % len(COLORS)]
            _add_single_trace(
                fig, seg_df, label=str(seg), color=color, row=row, col=col
            )


def _add_single_trace(
    fig: go.Figure,
    df: pl.DataFrame,
    label: str,
    color: str,
    row: int,
    col: int,
) -> None:
    """Add one mean line and its CI band to a subplot cell.

    Args:
        fig: Plotly Figure (created with make_subplots) to add traces to.
        df: Curve DataFrame for one series.
        label: Display name for the trace legend.
        color: Hex color for this series.
        row: Subplot row (1-indexed).
        col: Subplot column (1-indexed).
    """
    df = df.sort("horizon_value")
    horizons = df["horizon_value"].to_list()
    means = df["markout_mean"].to_list()
    uppers = df["markout_ci_upper"].to_list()
    lowers = df["markout_ci_lower"].to_list()

    # CI band: horizons forward + horizons reversed, upper + lower reversed
    band_x = horizons + list(reversed(horizons))
    band_y = uppers + list(reversed(lowers))
    fill_color = _hex_to_rgba(color, 0.15)

    fig.add_trace(
        go.Scatter(
            x=band_x,
            y=band_y,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
            name=f"{label} CI",
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=horizons,
            y=means,
            mode="lines+markers",
            name=label,
            line=dict(color=color),
            marker=dict(color=color, size=5),
        ),
        row=row,
        col=col,
    )


def plot_curve(result: MarkoutResult, *, by: str | None = None) -> go.Figure:
    """Plot the markout decay curve with bootstrap CI bands.

    Args:
        result: MarkoutResult instance to pull curve data from.
        by: Optional column name to segment traces by.

    Returns:
        Plotly Figure with one subplot per horizon_type.
    """
    curve_df = result.curve(by=by, n_bootstrap=500)

    horizon_types = curve_df["horizon_type"].unique(maintain_order=True).to_list()
    n_types = len(horizon_types)

    # Always use make_subplots so row/col are always valid.
    fig = make_subplots(
        rows=1,
        cols=n_types,
        subplot_titles=[str(t) for t in horizon_types] if n_types > 1 else None,
    )

    for col_idx, htype in enumerate(horizon_types):
        row = 1
        col = col_idx + 1
        sub_df = curve_df.filter(pl.col("horizon_type") == htype)

        _add_traces(fig, sub_df, by=by, row=row, col=col)

        # Horizontal reference line at y=0
        horizons_sorted = sub_df["horizon_value"].sort().to_list()
        x_min = horizons_sorted[0] if horizons_sorted else 0
        x_max = horizons_sorted[-1] if horizons_sorted else 1

        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
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

        # x-axis scale: log for wall-clock, linear otherwise
        # Plotly names axes xaxis, xaxis2, xaxis3, … for cols 1, 2, 3, …
        axis_key = "xaxis" if col_idx == 0 else f"xaxis{col_idx + 1}"
        if htype in _LOG_SCALE_TYPES:
            fig.update_layout(**{axis_key: dict(type="log")})

    apply_style(fig, "Markout Curve")
    return fig
