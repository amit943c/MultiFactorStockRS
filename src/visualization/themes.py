"""Plotly theme definitions for consistent, publication-quality charts."""

from __future__ import annotations

import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color palettes — professional, muted tones
# ---------------------------------------------------------------------------

PALETTE_MAIN: list[str] = [
    "#2E5C8A",  # steel blue
    "#3D8B8B",  # teal
    "#5B7F95",  # slate
    "#7BA7BC",  # sky
    "#4A7C59",  # forest
    "#8B6E4E",  # umber
    "#6B5B8A",  # muted purple
    "#A3785F",  # warm tan
]

PALETTE_ACCENT: list[str] = [
    "#C0392B",  # crimson (drawdown / negative)
    "#27AE60",  # emerald (positive)
    "#F39C12",  # amber (warning / highlight)
    "#2980B9",  # cerulean (secondary emphasis)
    "#8E44AD",  # deep purple (tertiary)
    "#16A085",  # turquoise
]

_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

# ---------------------------------------------------------------------------
# Theme dictionaries (Plotly layout overrides)
# ---------------------------------------------------------------------------

LIGHT_THEME: dict = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FAFBFC",
    "font": {
        "family": _FONT_FAMILY,
        "size": 13,
        "color": "#2C3E50",
    },
    "title": {
        "font": {"size": 18, "color": "#1A252F", "family": _FONT_FAMILY},
        "x": 0.5,
        "xanchor": "center",
    },
    "xaxis": {
        "title_font": {"size": 12, "color": "#5D6D7E"},
        "tickfont": {"size": 11, "color": "#5D6D7E"},
        "gridcolor": "#ECF0F1",
        "linecolor": "#D5DBDB",
        "zerolinecolor": "#D5DBDB",
        "showgrid": True,
    },
    "yaxis": {
        "title_font": {"size": 12, "color": "#5D6D7E"},
        "tickfont": {"size": 11, "color": "#5D6D7E"},
        "gridcolor": "#ECF0F1",
        "linecolor": "#D5DBDB",
        "zerolinecolor": "#D5DBDB",
        "showgrid": True,
    },
    "legend": {
        "bgcolor": "rgba(255,255,255,0.85)",
        "bordercolor": "#D5DBDB",
        "borderwidth": 1,
        "font": {"size": 11, "color": "#2C3E50"},
    },
    "colorway": PALETTE_MAIN,
    "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
}

DARK_THEME: dict = {
    "paper_bgcolor": "#1A1A2E",
    "plot_bgcolor": "#16213E",
    "font": {
        "family": _FONT_FAMILY,
        "size": 13,
        "color": "#E0E0E0",
    },
    "title": {
        "font": {"size": 18, "color": "#F5F5F5", "family": _FONT_FAMILY},
        "x": 0.5,
        "xanchor": "center",
    },
    "xaxis": {
        "title_font": {"size": 12, "color": "#B0BEC5"},
        "tickfont": {"size": 11, "color": "#B0BEC5"},
        "gridcolor": "#2A3A5C",
        "linecolor": "#2A3A5C",
        "zerolinecolor": "#37474F",
        "showgrid": True,
    },
    "yaxis": {
        "title_font": {"size": 12, "color": "#B0BEC5"},
        "tickfont": {"size": 11, "color": "#B0BEC5"},
        "gridcolor": "#2A3A5C",
        "linecolor": "#2A3A5C",
        "zerolinecolor": "#37474F",
        "showgrid": True,
    },
    "legend": {
        "bgcolor": "rgba(26,26,46,0.85)",
        "bordercolor": "#37474F",
        "borderwidth": 1,
        "font": {"size": 11, "color": "#E0E0E0"},
    },
    "colorway": [
        "#4FC3F7",  # bright sky
        "#81C784",  # soft green
        "#FFB74D",  # warm amber
        "#BA68C8",  # orchid
        "#4DD0E1",  # cyan
        "#FF8A65",  # soft coral
        "#AED581",  # lime
        "#90A4AE",  # blue-grey
    ],
    "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
}

_THEMES: dict[str, dict] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


def get_theme(name: str = "light") -> dict:
    """Return a theme dict by name (``"light"`` or ``"dark"``)."""
    key = name.lower().strip()
    if key not in _THEMES:
        raise ValueError(f"Unknown theme '{name}'. Choose from: {list(_THEMES)}")
    return _THEMES[key]


def get_plotly_template(name: str = "light") -> go.layout.Template:
    """Build a full :class:`plotly.graph_objects.layout.Template` from a theme."""
    theme = get_theme(name)
    template = go.layout.Template()
    template.layout = go.Layout(**theme)
    return template
