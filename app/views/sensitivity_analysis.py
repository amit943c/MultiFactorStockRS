"""Sensitivity Analysis page — parameter sweeps across top_n, costs, rebalance frequency."""

from __future__ import annotations

import copy
import logging

import pandas as pd
import streamlit as st

from src.analytics.research import SensitivityAnalyzer
from src.backtest.engine import BacktestEngine
from src.portfolio.construction import PortfolioConstructor
from src.portfolio.rebalance import RebalanceEngine
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)


def _check_prerequisites() -> tuple[bool, pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    """Return (ok, factor_df, prices_df, cfg)."""
    factor_df = st.session_state.get("factor_df")
    prices_df = st.session_state.get("prices_df")
    cfg = st.session_state.get("cfg")

    if factor_df is None or prices_df is None or cfg is None:
        return False, None, None, None
    return True, factor_df, prices_df, cfg


def render(theme: str) -> None:
    """Render the Sensitivity Analysis page."""
    st.header("Sensitivity Analysis")
    st.caption(
        "Parameter sweeps run the full portfolio → backtest pipeline for each setting. "
        "Click **Run Sweep** to generate results, or results will load from cache if available."
    )

    ok, factor_df, prices_df, cfg = _check_prerequisites()
    if not ok:
        st.info(
            "Sensitivity analysis requires factor data, price data, and a loaded config. "
            "Run the pipeline or load cached data first."
        )
        return

    sweep_tabs = st.tabs(["Top-N Sweep", "Transaction Cost Sweep", "Rebalance Frequency"])

    with sweep_tabs[0]:
        _render_top_n_sweep(factor_df, prices_df, cfg, theme)

    with sweep_tabs[1]:
        _render_cost_sweep(factor_df, prices_df, cfg, theme)

    with sweep_tabs[2]:
        _render_rebalance_sweep(factor_df, prices_df, cfg, theme)


def _render_top_n_sweep(
    factor_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cfg: dict,
    theme: str,
) -> None:
    st.subheader("Top-N Portfolio Size Sweep")

    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl:
        top_n_vals = st.multiselect(
            "Top-N values to test",
            options=[5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
            default=[5, 10, 15, 20, 30, 50],
            key="sa_top_n_vals",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_top_n = st.button("Run Sweep", key="sa_run_top_n", type="primary")

    if run_top_n and top_n_vals:
        with st.spinner("Running top-N sweep — this may take a minute…"):
            try:
                sweep_df = SensitivityAnalyzer.sweep_top_n(
                    factor_df, prices_df, cfg, top_n_values=sorted(top_n_vals),
                )
                st.session_state["sa_top_n_result"] = sweep_df
            except Exception:
                logger.exception("Top-N sweep failed")
                st.error("Top-N sweep failed. Check the logs for details.")
                return

    sweep_df = st.session_state.get("sa_top_n_result")
    if sweep_df is not None and len(sweep_df) > 0:
        tab_table, tab_chart = st.tabs(["Table", "Chart"])

        with tab_table:
            display_df = sweep_df.copy()
            st.dataframe(
                display_df.style.format({
                    "top_n": "{:.0f}",
                    "cagr": "{:.2%}",
                    "sharpe": "{:.3f}",
                    "max_dd": "{:.2%}",
                    "turnover": "{:.2%}",
                }),
                use_container_width=True,
            )

        with tab_chart:
            _render_sweep_metrics_chart(sweep_df, "top_n", "Top-N", theme)
    else:
        st.info("Click **Run Sweep** to generate top-N sensitivity results.")


def _render_cost_sweep(
    factor_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cfg: dict,
    theme: str,
) -> None:
    st.subheader("Transaction Cost Sweep")

    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl:
        cost_vals = st.multiselect(
            "Cost values (bps)",
            options=[0, 5, 10, 15, 20, 30, 50, 75, 100],
            default=[0, 5, 10, 20, 30, 50],
            key="sa_cost_vals",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_cost = st.button("Run Sweep", key="sa_run_cost", type="primary")

    if run_cost and cost_vals:
        with st.spinner("Running transaction cost sweep…"):
            try:
                sweep_df = SensitivityAnalyzer.sweep_transaction_costs(
                    factor_df, prices_df, cfg, cost_values=sorted(cost_vals),
                )
                st.session_state["sa_cost_result"] = sweep_df

                # Build before/after equity curves for 0 bps vs highest cost
                _compute_before_after_equity(factor_df, prices_df, cfg, max(cost_vals))
            except Exception:
                logger.exception("Transaction cost sweep failed")
                st.error("Transaction cost sweep failed.")
                return

    sweep_df = st.session_state.get("sa_cost_result")
    if sweep_df is not None and len(sweep_df) > 0:
        tab_table, tab_chart, tab_impact = st.tabs(["Table", "Chart", "Cost Impact"])

        with tab_table:
            st.dataframe(
                sweep_df.style.format({
                    "cost_bps": "{:.0f}",
                    "cagr": "{:.2%}",
                    "sharpe": "{:.3f}",
                    "max_dd": "{:.2%}",
                    "total_return": "{:.2%}",
                }),
                use_container_width=True,
            )

        with tab_chart:
            _render_sweep_metrics_chart(sweep_df, "cost_bps", "Cost (bps)", theme)

        with tab_impact:
            eq_gross = st.session_state.get("sa_equity_gross")
            eq_net = st.session_state.get("sa_equity_net")
            if eq_gross is not None and eq_net is not None:
                fig_ba = ChartFactory.before_after_costs_chart(
                    eq_gross, eq_net, theme=theme,
                )
                st.plotly_chart(fig_ba, use_container_width=True)
            else:
                st.info("Before/after equity data not available.")
    else:
        st.info("Click **Run Sweep** to generate transaction cost results.")


def _render_rebalance_sweep(
    factor_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cfg: dict,
    theme: str,
) -> None:
    st.subheader("Rebalance Frequency Comparison")

    col_ctrl, col_btn = st.columns([3, 1])
    with col_ctrl:
        freq_vals = st.multiselect(
            "Frequencies to test",
            options=["weekly", "biweekly", "monthly", "quarterly"],
            default=["weekly", "monthly"],
            key="sa_freq_vals",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_freq = st.button("Run Sweep", key="sa_run_freq", type="primary")

    if run_freq and freq_vals:
        with st.spinner("Running rebalance frequency sweep…"):
            try:
                sweep_df = SensitivityAnalyzer.sweep_rebalance_freq(
                    factor_df, prices_df, cfg, frequencies=freq_vals,
                )
                st.session_state["sa_freq_result"] = sweep_df
            except Exception:
                logger.exception("Rebalance frequency sweep failed")
                st.error("Rebalance frequency sweep failed.")
                return

    sweep_df = st.session_state.get("sa_freq_result")
    if sweep_df is not None and len(sweep_df) > 0:
        tab_table, tab_chart = st.tabs(["Table", "Chart"])

        with tab_table:
            st.dataframe(
                sweep_df.style.format({
                    "cagr": "{:.2%}",
                    "sharpe": "{:.3f}",
                    "max_dd": "{:.2%}",
                    "turnover": "{:.2%}",
                }),
                use_container_width=True,
            )

        with tab_chart:
            _render_sweep_metrics_chart(sweep_df, "frequency", "Frequency", theme)
    else:
        st.info("Click **Run Sweep** to generate rebalance frequency results.")


# ── Helpers ───────────────────────────────────────────────────────────


def _render_sweep_metrics_chart(
    sweep_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    theme: str,
) -> None:
    """Render a grouped bar chart of key metrics from a sweep DataFrame."""
    import plotly.graph_objects as go
    from src.visualization.themes import PALETTE_MAIN, get_theme

    metric_cols = [c for c in ["cagr", "sharpe", "max_dd", "turnover", "total_return"] if c in sweep_df.columns]
    if not metric_cols:
        st.info("No metrics available to chart.")
        return

    selected_metric = st.selectbox(
        "Metric to display",
        options=metric_cols,
        index=0,
        key=f"sa_metric_{x_col}",
    )

    x_vals = [str(v) for v in sweep_df[x_col]]
    y_vals = sweep_df[selected_metric].values

    is_pct = selected_metric in ("cagr", "max_dd", "turnover", "total_return")
    colors = [
        PALETTE_MAIN[0] if v >= 0 else "#C0392B" for v in y_vals
    ]

    fig = go.Figure(go.Bar(
        x=x_vals, y=y_vals,
        marker_color=colors,
        text=[f"{v:.2%}" if is_pct else f"{v:.3f}" for v in y_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{selected_metric.upper()} by {x_label}",
        xaxis_title=x_label,
        yaxis_title=selected_metric,
        yaxis_tickformat=".1%" if is_pct else ".2f",
    )
    fig.update_layout(**get_theme(theme))
    st.plotly_chart(fig, use_container_width=True)


def _compute_before_after_equity(
    factor_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cfg: dict,
    max_cost_bps: int,
) -> None:
    """Compute gross (0 bps) and net (max_cost_bps) equity curves for comparison."""
    try:
        cfg_gross = copy.deepcopy(cfg)
        cfg_gross.setdefault("portfolio", {})["transaction_cost_bps"] = 0

        constructor = PortfolioConstructor(cfg_gross)
        engine = RebalanceEngine(constructor)
        reb = engine.run(factor_df, prices_df, cfg_gross)
        bt_gross = BacktestEngine(cfg_gross).run(reb, prices_df)
        st.session_state["sa_equity_gross"] = bt_gross.equity_curve

        cfg_net = copy.deepcopy(cfg)
        cfg_net.setdefault("portfolio", {})["transaction_cost_bps"] = max_cost_bps
        constructor_net = PortfolioConstructor(cfg_net)
        engine_net = RebalanceEngine(constructor_net)
        reb_net = engine_net.run(factor_df, prices_df, cfg_net)
        bt_net = BacktestEngine(cfg_net).run(reb_net, prices_df)
        st.session_state["sa_equity_net"] = bt_net.equity_curve
    except Exception:
        logger.exception("Before/after equity computation failed")
