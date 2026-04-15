"""Research Visuals page — chart gallery, download buttons, and report display."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics.report import ReportGenerator
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render(theme: str) -> None:
    """Render the Research Visuals page."""
    st.header("Research Visuals")

    bt_result = st.session_state.get("backtest_result")
    factor_df: pd.DataFrame | None = st.session_state.get("factor_df")
    benchmark_equity: pd.Series | None = st.session_state.get("benchmark_equity")
    stats: dict | None = st.session_state.get("stats")

    if bt_result is None or len(bt_result.equity_curve) == 0:
        st.info("No backtest results available. Run the pipeline first.")
        return

    charts = _build_chart_gallery(bt_result, benchmark_equity, factor_df, theme)

    if not charts:
        st.warning("No charts could be generated from available data.")
        return

    # ── Gallery layout ───────────────────────────────────────────────
    st.subheader("Chart Gallery")
    cols_per_row = 2
    chart_items = list(charts.items())

    for i in range(0, len(chart_items), cols_per_row):
        row_items = chart_items[i : i + cols_per_row]
        cols = st.columns(len(row_items))
        for col, (name, fig) in zip(cols, row_items):
            with col:
                st.markdown(f"**{_format_chart_name(name)}**")
                st.plotly_chart(fig, use_container_width=True)
                _download_buttons(name, fig)

    # ── Summary report ───────────────────────────────────────────────
    if stats and bt_result:
        _render_report_section(stats, bt_result)


def _build_chart_gallery(
    bt_result,
    benchmark_equity: pd.Series | None,
    factor_df: pd.DataFrame | None,
    theme: str,
) -> dict[str, object]:
    """Build all available charts into a name→figure dict."""
    charts: dict[str, object] = {}

    if benchmark_equity is not None:
        try:
            charts["equity_curve"] = ChartFactory.equity_curve(
                bt_result.equity_curve, benchmark_equity, theme=theme,
            )
        except Exception:
            logger.exception("equity_curve failed")

    try:
        charts["rolling_drawdown"] = ChartFactory.rolling_drawdown(
            bt_result.equity_curve, theme=theme,
        )
    except Exception:
        logger.exception("rolling_drawdown failed")

    try:
        charts["rolling_sharpe"] = ChartFactory.rolling_sharpe_chart(
            bt_result.daily_returns, theme=theme,
        )
    except Exception:
        logger.exception("rolling_sharpe failed")

    try:
        charts["monthly_returns"] = ChartFactory.monthly_returns_heatmap(
            bt_result.daily_returns, theme=theme,
        )
    except Exception:
        logger.exception("monthly_returns_heatmap failed")

    if bt_result.turnover is not None and len(bt_result.turnover) > 0:
        try:
            charts["turnover"] = ChartFactory.turnover_over_time(
                bt_result.turnover, theme=theme,
            )
        except Exception:
            logger.exception("turnover chart failed")

    if factor_df is not None and len(factor_df) > 0:
        factor_cols = [
            c for c in factor_df.columns
            if c not in {"date", "ticker", "composite_score", "composite_rank"}
        ]
        if factor_cols:
            try:
                charts["factor_correlation"] = ChartFactory.factor_correlation_heatmap(
                    factor_df, factor_cols, theme=theme,
                )
            except Exception:
                logger.exception("factor_correlation failed")

        if "date" in factor_df.columns:
            last_date = factor_df["date"].max()
            try:
                charts["factor_distribution"] = ChartFactory.factor_score_distribution(
                    factor_df, last_date, theme=theme,
                )
            except Exception:
                logger.exception("factor_distribution failed")

            try:
                charts["top_ranked"] = ChartFactory.top_ranked_table(
                    factor_df, last_date, theme=theme,
                )
            except Exception:
                logger.exception("top_ranked failed")

    if bt_result.holdings_history and len(bt_result.holdings_history) > 1:
        try:
            charts["portfolio_composition"] = ChartFactory.portfolio_composition(
                bt_result.holdings_history, theme=theme,
            )
        except Exception:
            logger.exception("portfolio_composition failed")

    return charts


def _format_chart_name(name: str) -> str:
    """Convert snake_case chart key to a readable title."""
    return name.replace("_", " ").title()


def _download_buttons(name: str, fig) -> None:
    """Provide HTML download for a Plotly figure."""
    try:
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="Download HTML",
            data=html_bytes,
            file_name=f"{name}.html",
            mime="text/html",
            key=f"dl_html_{name}",
        )
    except Exception:
        logger.debug("HTML download button failed for %s", name)

    try:
        img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
        st.download_button(
            label="Download PNG",
            data=img_bytes,
            file_name=f"{name}.png",
            mime="image/png",
            key=f"dl_png_{name}",
        )
    except Exception:
        logger.debug("PNG download unavailable for %s (kaleido may not be installed)", name)


def _render_report_section(stats: dict, bt_result) -> None:
    """Display or generate the markdown summary report."""
    st.markdown("---")
    st.subheader("Summary Report")

    report_path = _PROJECT_ROOT / "outputs" / "reports" / "backtest_report.md"
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        with st.expander("View Report", expanded=False):
            st.markdown(content)
        st.download_button(
            label="Download Report (.md)",
            data=content.encode("utf-8"),
            file_name="backtest_report.md",
            mime="text/markdown",
            key="dl_report",
        )
    else:
        st.caption("No saved report found. Generate one by running the full pipeline.")

        if st.button("Generate Report Now", key="rv_gen_report"):
            cfg = st.session_state.get("cfg", {})
            out = str(report_path)
            try:
                ReportGenerator.generate_markdown(stats, bt_result, cfg, out)
                st.success(f"Report saved to `{out}`.")
                st.rerun()
            except Exception:
                logger.exception("Report generation failed")
                st.error("Failed to generate report.")
