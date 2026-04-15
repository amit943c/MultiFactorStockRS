"""Static and interactive chart export utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class ChartExporter:
    """Export Plotly figures to static images and HTML."""

    @staticmethod
    def save_figure(
        fig: go.Figure,
        path: str,
        format: str = "png",
        width: int = 1200,
        height: int = 700,
        scale: int = 2,
    ) -> None:
        """Write a figure to a static image file via kaleido.

        Parameters
        ----------
        fig:
            The Plotly figure to export.
        path:
            Destination file path (extension overridden by *format*).
        format:
            Image format — ``"png"``, ``"svg"``, ``"pdf"``, ``"jpeg"``.
        width, height:
            Image dimensions in pixels (before *scale* multiplier).
        scale:
            Resolution multiplier (2 = retina-quality).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(
            str(dest),
            format=format,
            width=width,
            height=height,
            scale=scale,
        )
        logger.info("Saved %s image → %s", format.upper(), dest)

    @staticmethod
    def save_html(fig: go.Figure, path: str) -> None:
        """Write the figure as a self-contained interactive HTML file."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(dest), include_plotlyjs="cdn")
        logger.info("Saved interactive HTML → %s", dest)

    @classmethod
    def save_all_charts(
        cls,
        charts: dict[str, go.Figure],
        output_dir: str,
        format: str = "png",
        width: int = 1200,
        height: int = 700,
        scale: int = 2,
    ) -> None:
        """Batch-export a name→figure mapping to *output_dir*.

        Each chart is saved as ``<output_dir>/<name>.<format>``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in charts.items():
            dest = out / f"{name}.{format}"
            cls.save_figure(fig, str(dest), format=format, width=width, height=height, scale=scale)
        logger.info("Batch export complete: %d charts → %s", len(charts), out)
