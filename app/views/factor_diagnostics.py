"""Factor Diagnostics page — correlation, IC, distributions, scatter, heatmaps."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.analytics.factor_analytics import FactorAnalytics
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)


def _factor_columns(factor_df: pd.DataFrame) -> list[str]:
    """Extract factor column names from the factor DataFrame."""
    skip = {"date", "ticker", "composite_score", "composite_rank"}
    return [c for c in factor_df.columns if c not in skip]


def _available_dates(factor_df: pd.DataFrame) -> list:
    """Return sorted unique dates from the factor DataFrame."""
    if "date" not in factor_df.columns:
        return []
    return sorted(factor_df["date"].unique())


def render(theme: str) -> None:
    """Render the Factor Diagnostics page."""
    st.header("Factor Diagnostics")

    factor_df: pd.DataFrame | None = st.session_state.get("factor_df")

    if factor_df is None or len(factor_df) == 0:
        st.info("No factor data available. Run the pipeline first.")
        return

    factor_cols = _factor_columns(factor_df)
    dates = _available_dates(factor_df)

    if not factor_cols:
        st.warning("No factor columns found in the data.")
        return

    # ── Date selector ────────────────────────────────────────────────
    selected_date = st.selectbox(
        "Select rebalance date",
        options=dates,
        index=len(dates) - 1 if dates else 0,
        key="fd_date",
    )

    # ── Factor correlation heatmap ───────────────────────────────────
    st.subheader("Factor Correlation Matrix")
    with st.spinner("Computing factor correlations..."):
        fig_corr = ChartFactory.factor_correlation_heatmap(
            factor_df, factor_cols, theme=theme,
        )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Factor score distribution ────────────────────────────────────
    if selected_date is not None:
        st.subheader(f"Composite Score Distribution — {selected_date}")
        top_n = st.slider(
            "Top N stocks to include", 10, 100, 50, step=10, key="fd_top_n",
        )
        fig_dist = ChartFactory.factor_score_distribution(
            factor_df, selected_date, top_n=top_n, theme=theme,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Factor IC summary ────────────────────────────────────────────
    _render_ic_summary(factor_df, factor_cols)

    # ── Factor heatmap for top stocks ────────────────────────────────
    if selected_date is not None:
        _render_factor_heatmap(factor_df, factor_cols, selected_date, theme)

    # ── Factor vs forward return scatter ─────────────────────────────
    _render_scatter(factor_df, factor_cols, theme)


def _render_ic_summary(factor_df: pd.DataFrame, factor_cols: list[str]) -> None:
    """Render the IC summary table if forward return columns are available."""
    return_cols = [c for c in factor_df.columns if c.startswith("return_")]
    if not return_cols:
        st.caption("No forward return columns available for IC analysis.")
        return

    st.subheader("Information Coefficient (IC) Summary")
    fwd_col = st.selectbox(
        "Forward return column",
        options=return_cols,
        index=0,
        key="fd_fwd_col",
    )

    try:
        with st.spinner("Computing IC summary..."):
            ic_summary = FactorAnalytics.factor_ic_summary(
                factor_df, factor_cols, forward_return_col=fwd_col,
            )
        st.dataframe(
            ic_summary.style.format({
                "mean_ic": "{:.4f}",
                "ic_std": "{:.4f}",
                "ic_ir": "{:.3f}",
                "hit_rate": "{:.1%}",
                "n_periods": "{:.0f}",
            }),
            use_container_width=True,
        )
    except Exception:
        logger.exception("IC summary computation failed")
        st.warning("Could not compute IC summary — data may be insufficient.")


def _render_factor_heatmap(
    factor_df: pd.DataFrame,
    factor_cols: list[str],
    selected_date: object,
    theme: str,
) -> None:
    """Factor z-score heatmap for the top-ranked stocks on the selected date."""
    st.subheader(f"Factor Z-Scores — Top Stocks on {selected_date}")
    mask = factor_df["date"] == pd.Timestamp(selected_date)
    date_slice = factor_df.loc[mask]

    if "composite_score" not in date_slice.columns or len(date_slice) == 0:
        st.caption("Insufficient data for factor heatmap.")
        return

    n_tickers = st.slider(
        "Number of tickers", 5, 30, 15, step=5, key="fd_hm_tickers",
    )
    top_tickers = (
        date_slice.nlargest(n_tickers, "composite_score")["ticker"].tolist()
    )

    fig_hm = ChartFactory.factor_heatmap(
        factor_df, selected_date, top_tickers, factor_cols, theme=theme,
    )
    st.plotly_chart(fig_hm, use_container_width=True)


def _render_scatter(
    factor_df: pd.DataFrame,
    factor_cols: list[str],
    theme: str,
) -> None:
    """Factor vs forward-return scatter plot."""
    return_cols = [c for c in factor_df.columns if c.startswith("return_")]
    if not return_cols:
        return

    st.subheader("Factor vs Forward Return")
    col_a, col_b = st.columns(2)
    with col_a:
        scatter_factor = st.selectbox(
            "Factor", options=factor_cols, index=0, key="fd_scatter_factor",
        )
    with col_b:
        scatter_return = st.selectbox(
            "Return column", options=return_cols, index=0, key="fd_scatter_ret",
        )

    try:
        fig_sc = ChartFactory.factor_vs_return_scatter(
            factor_df, scatter_factor, scatter_return, theme=theme,
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    except Exception:
        logger.exception("Scatter plot failed")
        st.warning("Could not render scatter — check that both columns contain valid data.")
