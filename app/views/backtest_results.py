"""Backtest Results page — equity curves, drawdown, Sharpe, heatmap, turnover."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.backtest.performance import PerformanceAnalyzer
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)


def render(theme: str) -> None:
    """Render the Backtest Results page."""
    st.header("Backtest Results")

    bt_result = st.session_state.get("backtest_result")
    benchmark_equity: pd.Series | None = st.session_state.get("benchmark_equity")
    stats: dict | None = st.session_state.get("stats")

    if bt_result is None or len(bt_result.equity_curve) == 0:
        st.info("No backtest results available. Run the pipeline first.")
        return

    # ── Equity curves (net vs gross, with benchmark) ─────────────────
    st.subheader("Equity Curves")
    curve_mode = st.radio(
        "Equity curve type",
        ["Net of Costs", "Gross", "Both"],
        horizontal=True,
        key="bt_curve_mode",
    )

    if benchmark_equity is not None:
        if curve_mode == "Net of Costs":
            fig = ChartFactory.equity_curve(
                bt_result.equity_curve, benchmark_equity,
                title="Portfolio (Net) vs Benchmark", theme=theme,
            )
        elif curve_mode == "Gross":
            fig = ChartFactory.equity_curve(
                bt_result.equity_curve_gross, benchmark_equity,
                title="Portfolio (Gross) vs Benchmark", theme=theme,
            )
        else:
            fig = ChartFactory.equity_curve(
                bt_result.equity_curve, benchmark_equity,
                title="Portfolio (Net) vs Benchmark", theme=theme,
            )
            fig.add_scatter(
                x=bt_result.equity_curve_gross.index,
                y=bt_result.equity_curve_gross.values,
                mode="lines",
                name="Portfolio (Gross)",
                line={"dash": "dash", "width": 1.5, "color": "#27AE60"},
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Benchmark equity not available.")

    # ── Rolling drawdown ─────────────────────────────────────────────
    st.subheader("Rolling Drawdown")
    fig_dd = ChartFactory.rolling_drawdown(bt_result.equity_curve, theme=theme)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Rolling Sharpe ───────────────────────────────────────────────
    st.subheader("Rolling Sharpe Ratio")
    sharpe_window = st.slider(
        "Rolling window (trading days)", 21, 252, 63, step=21, key="bt_sharpe_window",
    )
    fig_sharpe = ChartFactory.rolling_sharpe_chart(
        bt_result.daily_returns, window=sharpe_window, theme=theme,
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # ── Monthly returns heatmap ──────────────────────────────────────
    st.subheader("Monthly Returns Heatmap")
    fig_monthly = ChartFactory.monthly_returns_heatmap(
        bt_result.daily_returns, theme=theme,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ── Turnover over time ───────────────────────────────────────────
    if bt_result.turnover is not None and len(bt_result.turnover) > 0:
        st.subheader("Turnover at Rebalance")
        fig_turn = ChartFactory.turnover_over_time(bt_result.turnover, theme=theme)
        st.plotly_chart(fig_turn, use_container_width=True)

    # ── Detailed stats table ─────────────────────────────────────────
    if stats:
        with st.expander("Detailed Statistics", expanded=False):
            stats_df = PerformanceAnalyzer.format_stats(stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
