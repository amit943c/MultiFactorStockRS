"""Regime Analysis page — market regime classification and strategy performance by regime."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.research import (
    CalendarAnalyzer,
    DrawdownAnalyzer,
    RegimeAnalyzer,
)
from src.visualization.charts import ChartFactory
from src.visualization.themes import PALETTE_MAIN, get_theme

logger = logging.getLogger(__name__)

_REGIME_COLORS = {
    "bull_low_vol": "#27AE60",
    "bull_high_vol": "#F39C12",
    "bear_low_vol": "#3498DB",
    "bear_high_vol": "#C0392B",
    "unknown": "#7F8C8D",
}


def render(theme: str) -> None:
    """Render the Regime Analysis page."""
    st.header("Regime Analysis")

    bt_result = st.session_state.get("backtest_result")
    prices_df: pd.DataFrame | None = st.session_state.get("prices_df")
    cfg: dict | None = st.session_state.get("cfg")

    if bt_result is None or prices_df is None:
        st.info(
            "Regime analysis requires backtest results and price data. "
            "Run the pipeline or load cached data first."
        )
        return

    daily_returns = bt_result.daily_returns
    if daily_returns is None or len(daily_returns) == 0:
        st.warning("No daily return data available.")
        return

    regime_tabs = st.tabs(["Market Regimes", "Calendar Analysis", "Drawdowns"])

    bench_ticker = (cfg or {}).get("benchmark", {}).get("ticker", "SPY")
    bench_prices = _extract_benchmark_prices(prices_df, bench_ticker)

    # ── Tab 1: Market Regimes ─────────────────────────────────────────
    with regime_tabs[0]:
        st.subheader("Market Regime Classification")

        if bench_prices is None or len(bench_prices) < 250:
            st.warning(
                f"Insufficient benchmark price data for `{bench_ticker}`. "
                "Need at least 250 trading days for regime classification."
            )
        else:
            col_v, col_t = st.columns(2)
            with col_v:
                vol_window = st.slider(
                    "Volatility window (days)", 20, 120, 60, key="ra_vol_w",
                )
            with col_t:
                trend_window = st.slider(
                    "Trend MA window (days)", 50, 400, 200, key="ra_trend_w",
                )

            try:
                with st.spinner("Classifying market regimes..."):
                    regimes = RegimeAnalyzer.classify_regimes(
                        bench_prices, vol_window=vol_window, trend_window=trend_window,
                    )
                st.session_state["ra_regimes"] = regimes

                _render_regime_timeline(bench_prices, regimes, theme)

                st.subheader("Strategy Performance by Regime")
                try:
                    regime_stats = RegimeAnalyzer.regime_performance(daily_returns, regimes)
                    if len(regime_stats) > 0:
                        c_table, c_chart = st.columns([1, 1])
                        with c_table:
                            st.dataframe(
                                regime_stats.style.format({
                                    "ann_return": "{:.2%}",
                                    "ann_volatility": "{:.2%}",
                                    "sharpe": "{:.3f}",
                                    "max_drawdown": "{:.2%}",
                                    "pct_of_time": "{:.1f}%",
                                }),
                                use_container_width=True,
                            )
                        with c_chart:
                            fig_rp = ChartFactory.regime_performance_chart(
                                regime_stats, theme=theme,
                            )
                            st.plotly_chart(fig_rp, use_container_width=True)
                    else:
                        st.info("Could not compute regime performance — regimes may not overlap with returns.")
                except Exception:
                    logger.exception("Regime performance failed")
                    st.warning("Could not compute regime performance.")

                st.subheader("Regime Transition Matrix")
                try:
                    trans = RegimeAnalyzer.regime_transition_matrix(regimes)
                    st.dataframe(
                        trans.style.format("{:.2%}").background_gradient(
                            cmap="YlOrRd", axis=1,
                        ),
                        use_container_width=True,
                    )
                except Exception:
                    logger.exception("Transition matrix failed")
                    st.warning("Could not compute transition matrix.")

            except Exception:
                logger.exception("Regime classification failed")
                st.warning("Regime classification failed. The benchmark data may be insufficient.")

    # ── Tab 2: Calendar Analysis ──────────────────────────────────────
    with regime_tabs[1]:
        st.subheader("Calendar Analysis")

        cal_left, cal_right = st.columns(2)

        with cal_left:
            st.markdown("**Monthly Returns Heatmap**")
            try:
                fig_mr = ChartFactory.monthly_returns_heatmap(daily_returns, theme=theme)
                st.plotly_chart(fig_mr, use_container_width=True)
            except Exception:
                logger.exception("Monthly returns heatmap failed")
                st.warning("Could not generate monthly returns heatmap.")

        with cal_right:
            st.markdown("**Annual Returns**")
            try:
                yearly = CalendarAnalyzer.yearly_returns(daily_returns)
                if len(yearly) > 0:
                    fig_yr = ChartFactory.yearly_returns_bar(yearly, theme=theme)
                    st.plotly_chart(fig_yr, use_container_width=True)
                else:
                    st.info("Insufficient data for yearly returns.")
            except Exception:
                logger.exception("Yearly returns failed")
                st.warning("Could not compute yearly returns.")

        with st.expander("Monthly Statistics by Calendar Month", expanded=True):
            try:
                monthly_stats = CalendarAnalyzer.monthly_stats(daily_returns)
                if len(monthly_stats) > 0:
                    st.dataframe(
                        monthly_stats.style.format({
                            "avg_return": "{:.2%}",
                            "win_rate": "{:.1%}",
                            "best_year": "{:.0f}",
                            "worst_year": "{:.0f}",
                        }),
                        use_container_width=True,
                    )
            except Exception:
                logger.exception("Monthly stats failed")
                st.warning("Could not compute monthly statistics.")

        st.subheader("Best & Worst Periods")
        try:
            periods = CalendarAnalyzer.best_worst_periods(daily_returns, n=5)
            bw_left, bw_right = st.columns(2)

            with bw_left:
                st.markdown("**Best Months**")
                if len(periods["best_months"]) > 0:
                    df = periods["best_months"].copy()
                    df["return"] = df["return"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown("**Best Days**")
                if len(periods["best_days"]) > 0:
                    df = periods["best_days"].copy()
                    df["return"] = df["return"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True, hide_index=True)

            with bw_right:
                st.markdown("**Worst Months**")
                if len(periods["worst_months"]) > 0:
                    df = periods["worst_months"].copy()
                    df["return"] = df["return"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                st.markdown("**Worst Days**")
                if len(periods["worst_days"]) > 0:
                    df = periods["worst_days"].copy()
                    df["return"] = df["return"].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            logger.exception("Best/worst periods failed")
            st.warning("Could not compute best/worst periods.")

    # ── Tab 3: Drawdowns ──────────────────────────────────────────────
    with regime_tabs[2]:
        st.subheader("Drawdown Episodes")

        equity_curve = bt_result.equity_curve
        if equity_curve is not None and len(equity_curve) > 0:
            dd_threshold = st.slider(
                "Drawdown threshold (%)", -30, -1, -5, key="ra_dd_thresh",
            )

            try:
                episodes = DrawdownAnalyzer.find_episodes(
                    equity_curve, threshold=dd_threshold / 100.0,
                )
                if episodes:
                    episodes_df = DrawdownAnalyzer.episodes_to_dataframe(episodes)
                    fig_dd = ChartFactory.drawdown_episodes_table(episodes_df, theme=theme)
                    st.plotly_chart(fig_dd, use_container_width=True)

                    st.caption(f"Found {len(episodes)} drawdown episodes exceeding {dd_threshold}%.")
                else:
                    st.success(f"No drawdown episodes exceeding {dd_threshold}% found.")
            except Exception:
                logger.exception("Drawdown episode analysis failed")
                st.warning("Could not compute drawdown episodes.")

            with st.expander("Worst Drawdown Episodes", expanded=True):
                try:
                    worst = DrawdownAnalyzer.worst_episodes(equity_curve, top_n=5)
                    if len(worst) > 0:
                        st.dataframe(
                            worst.style.format({
                                "depth_pct": "{:.2f}%",
                                "duration_days": "{:.0f}",
                            }),
                            use_container_width=True,
                        )
                    else:
                        st.info("No significant drawdown episodes detected.")
                except Exception:
                    logger.exception("Worst episodes failed")
                    st.warning("Could not compute worst episodes.")
        else:
            st.info("No equity curve available for drawdown analysis.")


# ── Helpers ───────────────────────────────────────────────────────────


def _extract_benchmark_prices(
    prices_df: pd.DataFrame,
    ticker: str,
) -> pd.Series | None:
    """Extract a date-indexed adj_close Series for the benchmark ticker."""
    mask = prices_df["ticker"] == ticker
    subset = prices_df.loc[mask].copy()
    if len(subset) == 0:
        return None
    subset = subset.sort_values("date")
    subset.index = pd.DatetimeIndex(subset["date"])
    return subset["adj_close"]


def _render_regime_timeline(
    bench_prices: pd.Series,
    regimes: pd.Series,
    theme: str,
) -> None:
    """Render a colour-coded timeline of benchmark prices by regime."""
    fig = go.Figure()

    common_idx = bench_prices.index.intersection(regimes.index)
    if len(common_idx) == 0:
        st.info("No overlapping dates between benchmark prices and regime labels.")
        return

    prices_aligned = bench_prices.reindex(common_idx)
    regimes_aligned = regimes.reindex(common_idx)

    for regime_label, color in _REGIME_COLORS.items():
        mask = regimes_aligned == regime_label
        if not mask.any():
            continue

        masked_prices = prices_aligned.where(mask)
        fig.add_trace(go.Scatter(
            x=masked_prices.index,
            y=masked_prices.values,
            mode="markers",
            name=regime_label.replace("_", " ").title(),
            marker={"color": color, "size": 3, "opacity": 0.7},
        ))

    fig.add_trace(go.Scatter(
        x=prices_aligned.index,
        y=prices_aligned.values,
        mode="lines",
        name="Price",
        line={"color": "#7F8C8D", "width": 1, "dash": "dot"},
        opacity=0.5,
    ))

    fig.update_layout(
        title="Benchmark Price Colored by Regime",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.15},
    )
    fig.update_layout(**get_theme(theme))
    st.plotly_chart(fig, use_container_width=True)
