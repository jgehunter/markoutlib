"""Shared plot styling."""

COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]

DIVERGING_COLORSCALE = "RdBu_r"

LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)


def apply_style(fig, title: str | None = None) -> None:
    fig.update_layout(**LAYOUT_DEFAULTS)
    if title:
        fig.update_layout(title=dict(text=title, x=0.0, xanchor="left"))
