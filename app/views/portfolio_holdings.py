"""Portfolio Holdings page — ranked tables, composition, weights, sector exposure."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.analytics.portfolio_analytics import PortfolioAnalytics
from src.visualization.charts import ChartFactory

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _build_ranked_table(
    _factor_df: pd.DataFrame, date_str: str, top_n: int
) -> tuple[pd.DataFrame, dict[str, str], int]:
    """Filter and format the ranked table (cached to avoid recomputation)."""
    date_ts = pd.Timestamp(date_str)
    mask = _factor_df["date"] == date_ts
    total = int(mask.sum())
    ranked = _factor_df.loc[mask].nlargest(top_n, "composite_score")

    display_cols = [
        c for c in ["ticker", "composite_score", "composite_rank",
                     "return_1m", "return_3m", "return_6m",
                     "dist_ma50", "dist_ma200"]
        if c in ranked.columns
    ]
    display_df = ranked[display_cols].copy()

    fmt: dict[str, str] = {}
    for c in display_df.columns:
        if c in ("composite_score",):
            fmt[c] = "{:.3f}"
        elif c in ("composite_rank",):
            fmt[c] = "{:.0f}"
        elif c.startswith("return_") or c.startswith("dist_"):
            fmt[c] = "{:.2%}"

    return display_df, fmt, total


def render(theme: str) -> None:
    """Render the Portfolio Holdings page."""
    st.header("Portfolio Holdings")

    bt_result = st.session_state.get("backtest_result")
    factor_df: pd.DataFrame | None = st.session_state.get("factor_df")

    if bt_result is None or not bt_result.holdings_history:
        st.info("No holdings data available. Run the pipeline first.")
        return

    holdings_history = bt_result.holdings_history
    rebalance_dates = sorted(holdings_history.keys())

    # ── Date selector ────────────────────────────────────────────────
    selected_date = st.selectbox(
        "Rebalance date",
        options=rebalance_dates,
        index=len(rebalance_dates) - 1,
        key="ph_date",
    )

    holdings = holdings_history.get(selected_date)
    if holdings is None or len(holdings) == 0:
        st.warning(f"No holdings for {selected_date}.")
        return

    # ── Top ranked stocks table ──────────────────────────────────────
    if factor_df is not None and len(factor_df) > 0:
        st.subheader(f"Top Ranked Stocks — {selected_date}")
        top_n = st.slider(
            "Display top N", 5, 100, 50, step=5, key="ph_top_n",
        )
        with st.spinner(f"Ranking top {top_n} stocks..."):
            try:
                display_df, fmt, total = _build_ranked_table(
                    factor_df, str(selected_date), top_n,
                )
                st.dataframe(
                    display_df.style.format(fmt, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                    height=min(len(display_df) * 40 + 40, 600),
                )
                st.caption(f"Showing {len(display_df)} of {total} stocks in universe.")
            except Exception:
                logger.exception("Top-ranked table render failed")
                st.warning("Could not render ranked table for this date.")

    # ── Holdings detail table (plain dataframe) ──────────────────────
    st.subheader(f"Holdings Detail — {selected_date}")
    display = holdings.copy()
    if "weight" in display.columns:
        display = display.sort_values("weight", ascending=False)
        display["weight_pct"] = display["weight"].map(lambda w: f"{w:.2%}")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Weight distribution stats ────────────────────────────────────
    st.subheader("Weight Distribution")
    try:
        dist_stats = PortfolioAnalytics.weight_distribution_stats(holdings)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions", dist_stats["n_positions"])
        c2.metric("Effective N", f"{dist_stats['effective_n']:.1f}")
        c3.metric("Top-5 Conc.", f"{dist_stats['top5_concentration']:.1%}")
        c4.metric("Max Weight", f"{dist_stats['max']:.2%}")

        with st.expander("Full Weight Statistics"):
            stats_rows = [
                {"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                for k, v in dist_stats.items()
            ]
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)
    except Exception:
        logger.exception("Weight distribution stats failed")

    # ── Portfolio composition (stacked bar) ──────────────────────────
    if len(rebalance_dates) > 1:
        st.subheader("Portfolio Composition Over Time")
        with st.spinner("Building composition chart..."):
            fig_comp = ChartFactory.portfolio_composition(
                holdings_history, theme=theme,
            )
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── Sector exposure ──────────────────────────────────────────────
    _render_sector_exposure(holdings_history, theme)


def _render_sector_exposure(holdings_history: dict, theme: str) -> None:
    """Render sector exposure if fundamentals data is available in session state."""
    fundamentals_df = st.session_state.get("fundamentals_df")
    if fundamentals_df is None:
        st.caption(
            "Sector exposure chart requires fundamentals data "
            "(not currently available — enable the fundamental factor group in config)."
        )
        return

    st.subheader("Sector Exposure Over Time")
    try:
        fig_sector = ChartFactory.sector_exposure(
            holdings_history, fundamentals_df, theme=theme,
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    except Exception:
        logger.exception("Sector exposure chart failed")
        st.warning("Could not render sector exposure chart.")
