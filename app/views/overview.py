"""Overview page — key metrics, equity curve, and config summary."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.backtest.performance import PerformanceAnalyzer
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)


def render(theme: str) -> None:
    """Render the Overview page using data from session state."""
    st.header("Strategy Overview")

    stats: dict | None = st.session_state.get("stats")
    bt_result = st.session_state.get("backtest_result")
    benchmark_equity: pd.Series | None = st.session_state.get("benchmark_equity")
    cfg: dict | None = st.session_state.get("cfg")

    if stats is None or bt_result is None:
        st.info(
            "No backtest results available yet. "
            "Click **Run Pipeline** in the sidebar, or ensure cached data exists in `data/processed/`."
        )
        return

    # ── Key metric cards ─────────────────────────────────────────────
    _render_metric_cards(stats)
    st.markdown("---")

    # ── Equity curve ─────────────────────────────────────────────────
    if benchmark_equity is not None and len(bt_result.equity_curve) > 0:
        fig = ChartFactory.equity_curve(
            bt_result.equity_curve, benchmark_equity, theme=theme,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Performance stats table ──────────────────────────────────────
    with st.expander("Detailed Performance Statistics", expanded=True):
        stats_df = PerformanceAnalyzer.format_stats(stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── Config summary ───────────────────────────────────────────────
    if cfg:
        _render_config_summary(cfg)


def _render_metric_cards(stats: dict) -> None:
    """Display the four headline metrics in a column layout."""
    cagr = stats.get("cagr", 0.0)
    sharpe = stats.get("sharpe_ratio", 0.0)
    max_dd = stats.get("max_drawdown", 0.0)
    total_ret = stats.get("total_return", 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CAGR", f"{cagr * 100:+.2f}%")
    c2.metric("Sharpe Ratio", f"{sharpe:.3f}")
    c3.metric("Max Drawdown", f"{max_dd * 100:+.2f}%")
    c4.metric("Total Return", f"{total_ret * 100:+.2f}%")


def _render_config_summary(cfg: dict) -> None:
    """Collapsible config summary section."""
    with st.expander("Configuration Summary"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Universe & Dates")
            uni = cfg.get("universe", {})
            dates = cfg.get("dates", {})
            st.markdown(f"- **Source**: `{uni.get('source', 'N/A')}`")
            st.markdown(f"- **Min history**: {uni.get('min_history_days', 'N/A')} days")
            st.markdown(f"- **Date range**: {dates.get('start', '?')} → {dates.get('end', '?')}")

            st.subheader("Portfolio")
            port = cfg.get("portfolio", {})
            st.markdown(f"- **Top N**: {port.get('top_n', 'N/A')}")
            st.markdown(f"- **Max weight**: {port.get('max_position_weight', 'N/A')}")
            st.markdown(f"- **Equal weight**: {port.get('equal_weight', 'N/A')}")
            st.markdown(f"- **TC (bps)**: {port.get('transaction_cost_bps', 'N/A')}")

        with col_b:
            st.subheader("Rebalance")
            reb = cfg.get("rebalance", {})
            st.markdown(f"- **Frequency**: {reb.get('frequency', 'N/A')}")
            st.markdown(f"- **Day of week**: {reb.get('day_of_week', 'N/A')}")

            st.subheader("Active Factors")
            for group, meta in cfg.get("factors", {}).items():
                if meta.get("enabled"):
                    subs = list(meta.get("weights", {}).keys())
                    st.markdown(f"- **{group}**: {', '.join(subs)}")
