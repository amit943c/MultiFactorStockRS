"""Research Diagnostics page — factor IC, quantile analysis, score persistence, stability."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.analytics.factor_analytics import FactorAnalytics
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)


def _factor_columns(factor_df: pd.DataFrame) -> list[str]:
    skip = {"date", "ticker", "composite_score", "composite_rank"}
    return [c for c in factor_df.columns if c not in skip]


def _return_columns(factor_df: pd.DataFrame) -> list[str]:
    return [c for c in factor_df.columns if c.startswith("return_")]


def render(theme: str) -> None:
    """Render the Research Diagnostics page."""
    st.header("Research Diagnostics")

    factor_df: pd.DataFrame | None = st.session_state.get("factor_df")
    bt_result = st.session_state.get("backtest_result")

    if factor_df is None or len(factor_df) == 0:
        st.info("No factor data available. Run the pipeline or load cached data first.")
        return

    factor_cols = _factor_columns(factor_df)
    return_cols = _return_columns(factor_df)

    if not factor_cols:
        st.warning("No factor columns found in the data.")
        return

    all_factor_choices = (
        ["composite_score"] if "composite_score" in factor_df.columns else []
    ) + factor_cols

    # ── Controls ──────────────────────────────────────────────────────
    ctrl_left, ctrl_mid, ctrl_right = st.columns(3)
    with ctrl_left:
        selected_factor = st.selectbox(
            "Factor",
            options=all_factor_choices,
            index=0,
            key="rd_factor",
        )
    with ctrl_mid:
        fwd_return_col = st.selectbox(
            "Forward return column",
            options=return_cols if return_cols else ["return_1m"],
            index=0,
            key="rd_fwd_ret",
        )
    with ctrl_right:
        n_quantiles = st.slider(
            "Quantiles", min_value=3, max_value=10, value=5, key="rd_nq",
        )

    if not return_cols:
        st.warning(
            "No forward return columns (e.g. `return_1m`) found in factor data. "
            "Some charts require forward returns."
        )

    rd_tabs = st.tabs([
        "IC Analysis",
        "Quantile Returns",
        "Stability & Overlap",
        "Factor Decay",
    ])

    # ── Tab 1: IC Analysis ────────────────────────────────────────────
    with rd_tabs[0]:
        if return_cols:
            st.subheader("Rolling Information Coefficient")
            window = st.slider(
                "Rolling window (periods)", 6, 52, 12, key="rd_ic_window",
            )
            with st.spinner("Computing rolling IC..."):
                try:
                    ic_series = FactorAnalytics.rolling_ic(
                        factor_df, selected_factor, fwd_return_col, window=window,
                    )
                    if len(ic_series) > 0:
                        fig_ic = ChartFactory.rolling_ic_chart(
                            ic_series, factor_name=selected_factor, theme=theme,
                        )
                        st.plotly_chart(fig_ic, use_container_width=True)

                        ic_all = FactorAnalytics.factor_ic(
                            factor_df, selected_factor, fwd_return_col,
                        )
                        ic_clean = ic_all.dropna()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Mean IC", f"{ic_clean.mean():.4f}")
                        c2.metric("IC Std", f"{ic_clean.std():.4f}")
                        ir = ic_clean.mean() / ic_clean.std() if ic_clean.std() > 1e-12 else 0.0
                        c3.metric("IC IR", f"{ir:.3f}")
                        c4.metric("Hit Rate", f"{(ic_clean > 0).mean():.1%}")
                    else:
                        st.info("Insufficient data for rolling IC with the selected window.")
                except Exception:
                    logger.exception("Rolling IC computation failed")
                    st.warning("Could not compute rolling IC — data may be insufficient.")
        else:
            st.info("Forward return columns required for IC analysis.")

    # ── Tab 2: Quantile Returns ───────────────────────────────────────
    with rd_tabs[1]:
        if return_cols:
            st.subheader("Quantile Cumulative Returns")
            with st.spinner("Computing quantile returns..."):
                try:
                    q_cum = FactorAnalytics.quantile_cumulative_returns(
                        factor_df, selected_factor, fwd_return_col, n_quantiles=n_quantiles,
                    )
                    if len(q_cum) > 0:
                        fig_q = ChartFactory.quantile_returns_chart(q_cum, theme=theme)
                        st.plotly_chart(fig_q, use_container_width=True)
                    else:
                        st.info("Insufficient data for quantile returns.")
                except Exception:
                    logger.exception("Quantile cumulative returns failed")
                    st.warning("Could not compute quantile returns.")

            st.subheader("Long-Short Spread (Top vs Bottom Quantile)")
            with st.spinner("Computing long-short spread..."):
                try:
                    spread = FactorAnalytics.long_short_spread(
                        factor_df, selected_factor, fwd_return_col, n_quantiles=n_quantiles,
                    )
                    if len(spread) > 0:
                        fig_ls = ChartFactory.long_short_spread_chart(spread, theme=theme)
                        st.plotly_chart(fig_ls, use_container_width=True)
                    else:
                        st.info("Insufficient data for long-short spread.")
                except Exception:
                    logger.exception("Long-short spread failed")
                    st.warning("Could not compute long-short spread.")
        else:
            st.info("Forward return columns required for quantile analysis.")

    # ── Tab 3: Stability & Overlap ────────────────────────────────────
    with rd_tabs[2]:
        st.subheader("Score Persistence (Rank Autocorrelation)")
        try:
            with st.spinner("Computing persistence..."):
                persistence = FactorAnalytics.score_persistence(factor_df, factor_col=selected_factor)
            if len(persistence) > 0:
                import plotly.graph_objects as go
                from src.visualization.themes import PALETTE_MAIN, get_theme

                fig_sp = go.Figure()
                fig_sp.add_trace(go.Scatter(
                    x=persistence.index, y=persistence.values,
                    mode="lines", name="Persistence",
                    line={"color": PALETTE_MAIN[0], "width": 2},
                ))
                fig_sp.add_hline(
                    y=persistence.mean(), line_dash="dash", line_color="#7F8C8D",
                    annotation_text=f"mean {persistence.mean():.3f}",
                    annotation_position="top right",
                )
                fig_sp.update_layout(
                    title=f"Score Persistence — {selected_factor}",
                    xaxis_title="Date", yaxis_title="Rank Correlation",
                    hovermode="x unified",
                )
                fig_sp.update_layout(**get_theme(theme))
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                st.info("Insufficient data for persistence analysis.")
        except Exception:
            logger.exception("Score persistence failed")
            st.warning("Could not compute score persistence.")

        if bt_result is not None and hasattr(bt_result, "holdings_history") and bt_result.holdings_history:
            st.subheader("Holdings Overlap (Jaccard Similarity)")
            try:
                hh = bt_result.holdings_history
                holdings_sets: dict[str, set[str]] = {}
                for dt, hdf in hh.items():
                    tickers = set(hdf.loc[hdf["ticker"] != "_CASH", "ticker"].tolist())
                    holdings_sets[str(dt)] = tickers

                if len(holdings_sets) > 1:
                    overlap = FactorAnalytics.holdings_overlap(holdings_sets)
                    if len(overlap) > 0:
                        fig_ov = ChartFactory.holdings_overlap_chart(overlap, theme=theme)
                        st.plotly_chart(fig_ov, use_container_width=True)
                    else:
                        st.info("Not enough rebalance periods for overlap analysis.")
                else:
                    st.info("Need at least 2 rebalance periods for overlap analysis.")
            except Exception:
                logger.exception("Holdings overlap failed")
                st.warning("Could not compute holdings overlap.")

        st.subheader("Score Dispersion Over Time")
        try:
            score_col = selected_factor if selected_factor in factor_df.columns else "composite_score"
            dispersion = FactorAnalytics.ranking_dispersion(factor_df, score_col=score_col)
            if len(dispersion) > 0:
                fig_disp = ChartFactory.score_dispersion_chart(dispersion, theme=theme)
                st.plotly_chart(fig_disp, use_container_width=True)
            else:
                st.info("Insufficient data for dispersion analysis.")
        except Exception:
            logger.exception("Score dispersion failed")
            st.warning("Could not compute score dispersion.")

        st.subheader("Top Holdings Stability")
        top_n_stab = st.slider(
            "Top N for stability", 5, 50, 20, step=5, key="rd_stab_n",
        )
        try:
            score_col = "composite_score" if "composite_score" in factor_df.columns else selected_factor
            stability = FactorAnalytics.top_holdings_stability(
                factor_df, top_n=top_n_stab, score_col=score_col,
            )
            if len(stability) > 0:
                import plotly.graph_objects as go
                from src.visualization.themes import PALETTE_MAIN, get_theme

                fig_stab = go.Figure()
                fig_stab.add_trace(go.Scatter(
                    x=stability.index, y=stability.values,
                    mode="lines+markers", name="Retention Rate",
                    line={"color": PALETTE_MAIN[0], "width": 2},
                    marker={"size": 4},
                ))
                fig_stab.add_hline(
                    y=stability.mean(), line_dash="dash", line_color="#7F8C8D",
                    annotation_text=f"mean {stability.mean():.2%}",
                    annotation_position="top right",
                )
                fig_stab.update_layout(
                    title=f"Top-{top_n_stab} Retention Rate",
                    xaxis_title="Date", yaxis_title="Retention Rate",
                    yaxis_tickformat=".0%", hovermode="x unified",
                )
                fig_stab.update_layout(**get_theme(theme))
                st.plotly_chart(fig_stab, use_container_width=True)
            else:
                st.info("Insufficient data for stability analysis.")
        except Exception:
            logger.exception("Top holdings stability failed")
            st.warning("Could not compute top holdings stability.")

    # ── Tab 4: Factor Decay ───────────────────────────────────────────
    with rd_tabs[3]:
        st.subheader("Factor Decay Analysis")
        max_lag = st.slider("Max lag (periods)", 3, 20, 10, key="rd_decay_lag")
        try:
            with st.spinner("Computing factor decay..."):
                decay_df = FactorAnalytics.factor_decay(factor_df, selected_factor, max_lag=max_lag)
            if len(decay_df) > 0 and not decay_df["ic"].isna().all():
                import plotly.graph_objects as go
                from src.visualization.themes import PALETTE_MAIN, get_theme

                fig_decay = go.Figure()
                fig_decay.add_trace(go.Bar(
                    x=[str(lag) for lag in decay_df.index],
                    y=decay_df["ic"].values,
                    marker_color=PALETTE_MAIN[0],
                ))
                fig_decay.add_hline(y=0, line_dash="dash", line_color="#7F8C8D")
                fig_decay.update_layout(
                    title=f"Factor Decay — {selected_factor}",
                    xaxis_title="Lag (periods)", yaxis_title="Average IC",
                    hovermode="x unified",
                )
                fig_decay.update_layout(**get_theme(theme))
                st.plotly_chart(fig_decay, use_container_width=True)

                with st.expander("Decay Data"):
                    st.dataframe(decay_df.style.format({"ic": "{:.4f}"}), use_container_width=True)
            else:
                st.info("Insufficient data for factor decay analysis.")
        except Exception:
            logger.exception("Factor decay failed")
            st.warning("Could not compute factor decay.")
