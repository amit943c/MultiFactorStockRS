"""Multi-Factor Stock RS — Streamlit Dashboard.

Launch with::

    streamlit run app/dashboard.py

The dashboard reads cached results from ``data/processed/`` (parquet) and
``outputs/`` directories, or can trigger a fresh pipeline run via the sidebar.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Ensure project root is on sys.path so ``src.*`` imports work ─────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.backtest.benchmark import BenchmarkTracker
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceAnalyzer
from src.data.store import DataStore
from src.factors.registry import FactorRegistry
from src.portfolio.construction import PortfolioConstructor
from src.portfolio.rebalance import RebalanceEngine
from src.utils.config import load_config

from app.views import (  # noqa: E402 — path already fixed above
    assumptions_limitations,
    backtest_results,
    factor_diagnostics,
    overview,
    portfolio_holdings,
    regime_analysis,
    research_diagnostics,
    research_visuals,
    sensitivity_analysis,
)

logger = logging.getLogger(__name__)

# =====================================================================
# Page configuration — MUST be the first Streamlit command
# =====================================================================

st.set_page_config(
    page_title="Multi-Factor Stock RS",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS for professional appearance
# =====================================================================

_CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f5f5f5;
    }
    section[data-testid="stSidebar"] label {
        color: #b0bec5 !important;
    }

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        font-size: 0.85rem;
        color: #5d6d7e;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 700;
        font-size: 1.6rem;
        color: #1a252f;
    }

    /* Dark-mode metric override */
    .dark-mode div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2a3a 0%, #16213e 100%);
        border-color: #2a3a5c;
    }
    .dark-mode div[data-testid="stMetric"] label {
        color: #b0bec5;
    }
    .dark-mode div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f5f5f5;
    }

    /* Clean header */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Expander styling */
    details {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0;
    }

    /* Dataframe container */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Button refinement */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }

    /* Navigation radio — styled as tab bar */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 2px;
        flex-wrap: nowrap;
        border-bottom: 2px solid #dee2e6;
        padding-bottom: 0;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        padding: 8px 14px;
        font-weight: 500;
        font-size: 0.8rem;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.15s ease;
        margin-bottom: -2px;
        color: #6c757d;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label p {
        font-size: 0.8rem !important;
        margin: 0;
    }
    /* Hide the radio circle */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        color: #667eea;
        border-bottom-color: rgba(102, 126, 234, 0.4);
    }
    /* Active / selected tab — multiple selectors for compatibility */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-baseweb="radio"]:has(input:checked),
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        color: #667eea;
        font-weight: 700;
        border-bottom: 3px solid #667eea;
        background: rgba(102, 126, 234, 0.06);
    }

    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: #2a3a5c;
    }
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================================
# Constants
# =====================================================================

_PAGES: list[tuple[str, str, object]] = [
    ("Overview", "overview", overview),
    ("Backtest", "backtest", backtest_results),
    ("Factors", "factors", factor_diagnostics),
    ("Holdings", "holdings", portfolio_holdings),
    ("Visuals", "visuals", research_visuals),
    ("Research", "research", research_diagnostics),
    ("Sensitivity", "sensitivity", sensitivity_analysis),
    ("Regimes", "regimes", regime_analysis),
    ("Assumptions", "assumptions", assumptions_limitations),
]

_PAGE_LOOKUP: dict[str, object] = {label: mod for label, _, mod in _PAGES}

# =====================================================================
# Session state initialisation
# =====================================================================

_STATE_DEFAULTS: dict[str, Any] = {
    "cfg": None,
    "prices_df": None,
    "factor_df": None,
    "backtest_result": None,
    "benchmark_equity": None,
    "stats": None,
    "fundamentals_df": None,
    "pipeline_ran": False,
    "config_path": "config/default_config.yaml",
    "theme_name": "light",
}

for key, default in _STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================================
# Sidebar
# =====================================================================

def _render_sidebar() -> None:
    """Build the sidebar with controls (no navigation — tabs are in the main area)."""
    with st.sidebar:
        st.markdown("# 📈 Multi-Factor Stock RS")
        st.caption("Quantitative Portfolio Dashboard")
        st.markdown("---")

        # Theme toggle
        theme_choice = st.toggle("Dark theme", value=st.session_state["theme_name"] == "dark")
        st.session_state["theme_name"] = "dark" if theme_choice else "light"

        if theme_choice:
            st.markdown(
                '<script>document.body.classList.add("dark-mode");</script>',
                unsafe_allow_html=True,
            )

        # Config path
        config_path = st.text_input(
            "Config file",
            value=st.session_state["config_path"],
            key="sidebar_config_path",
        )
        st.session_state["config_path"] = config_path

        st.markdown("---")

        # Run Pipeline button
        if st.button("🚀  Run Pipeline", use_container_width=True, type="primary"):
            _run_pipeline()

        # Load from cache button
        if not st.session_state.get("pipeline_ran"):
            if st.button("📂  Load Cached Data", use_container_width=True):
                _load_cached_data()

        # Config / data summary
        st.markdown("---")
        _render_sidebar_info()


def _render_sidebar_info() -> None:
    """Display config summary and data availability in the sidebar."""
    cfg = st.session_state.get("cfg")
    if cfg:
        dates = cfg.get("dates", {})
        st.markdown(f"**Date range**: `{dates.get('start', '?')}` → `{dates.get('end', '?')}`")
        prices = st.session_state.get("prices_df")
        if prices is not None:
            n_tickers = prices["ticker"].nunique() if "ticker" in prices.columns else 0
            st.markdown(f"**Universe**: {n_tickers} tickers")
    else:
        st.caption("No config loaded yet.")

    bt = st.session_state.get("backtest_result")
    if bt is not None and len(bt.equity_curve) > 0:
        st.success("Pipeline data loaded", icon="✅")
    else:
        st.caption("Awaiting pipeline run or cached data.")


# =====================================================================
# Data loading & pipeline execution
# =====================================================================

@st.cache_data(show_spinner=False)
def _load_config_cached(path: str) -> dict[str, Any]:
    return load_config(path)


def _merge_benchmark(prices_df: pd.DataFrame, store: DataStore, cfg: dict[str, Any]) -> pd.DataFrame:
    """Ensure benchmark ticker is present in prices_df by merging cached benchmark data."""
    bench_ticker = cfg.get("benchmark", {}).get("ticker", "SPY")
    if bench_ticker in prices_df["ticker"].unique():
        return prices_df
    if store.has_cache("benchmark_prices"):
        bp = store.load_prices("benchmark_prices")
        prices_df = pd.concat([prices_df, bp], ignore_index=True)
        logger.info("Merged cached benchmark %s into prices (%d rows)", bench_ticker, len(bp))
    return prices_df


def _load_cached_data() -> None:
    """Attempt to load pre-computed data from the DataStore cache."""
    config_path = st.session_state["config_path"]

    with st.spinner("Loading configuration…"):
        try:
            cfg = _load_config_cached(config_path)
            st.session_state["cfg"] = cfg
        except FileNotFoundError:
            st.error(f"Config file not found: `{config_path}`")
            return

    store = DataStore()

    # Prices
    if store.has_cache("prices"):
        with st.spinner("Loading cached prices…"):
            prices_df = store.load_prices("prices")
            prices_df = _merge_benchmark(prices_df, store, cfg)
            st.session_state["prices_df"] = prices_df
    else:
        st.warning("No cached price data found in `data/processed/prices.parquet`.")
        return

    # Factors
    if store.has_cache("factors"):
        with st.spinner("Loading cached factors…"):
            st.session_state["factor_df"] = store.load_factors("factors")

    prices_df = st.session_state["prices_df"]
    factor_df = st.session_state.get("factor_df")

    if factor_df is None or prices_df is None:
        st.warning("Cached factor data not available — run the pipeline for full results.")
        return

    # Reconstruct backtest from cached data
    with st.spinner("Reconstructing backtest from cached data…"):
        try:
            _run_backtest_from_cached(cfg, prices_df, factor_df)
        except Exception as exc:
            logger.exception("Failed to reconstruct backtest from cache")
            st.error(f"Backtest reconstruction failed: {exc}")
            return

    st.session_state["pipeline_ran"] = True
    st.toast("Cached data loaded successfully!", icon="✅")
    st.rerun()


def _run_backtest_from_cached(
    cfg: dict[str, Any],
    prices_df: pd.DataFrame,
    factor_df: pd.DataFrame,
) -> None:
    """Run portfolio construction + backtest using already-computed factors."""
    constructor = PortfolioConstructor(cfg)
    reb_engine = RebalanceEngine(constructor)
    rebalance_history = reb_engine.run(factor_df, prices_df, cfg)

    bt_engine = BacktestEngine(cfg)
    bt_result = bt_engine.run(rebalance_history, prices_df)
    st.session_state["backtest_result"] = bt_result

    # Benchmark
    bench_ticker = cfg.get("benchmark", {}).get("ticker", "SPY")
    dates_cfg = cfg.get("dates", {})
    tracker = BenchmarkTracker(bench_ticker)
    benchmark_equity = tracker.compute(
        prices_df, str(dates_cfg["start"]), str(dates_cfg["end"]),
    )
    st.session_state["benchmark_equity"] = benchmark_equity

    # Stats
    stats = PerformanceAnalyzer.compute_stats(
        equity_curve=bt_result.equity_curve,
        daily_returns=bt_result.daily_returns,
        benchmark_equity=benchmark_equity,
        turnover=bt_result.turnover,
    )
    st.session_state["stats"] = stats


def _run_pipeline() -> None:
    """Execute the full pipeline: fetch → factors → backtest."""
    config_path = st.session_state["config_path"]

    try:
        cfg = _load_config_cached(config_path)
        st.session_state["cfg"] = cfg
    except FileNotFoundError:
        st.error(f"Config file not found: `{config_path}`")
        return

    store = DataStore()
    progress = st.progress(0, text="Initialising pipeline…")

    # ── 1. Load universe ─────────────────────────────────────────────
    progress.progress(5, text="Loading ticker universe…")
    try:
        from src.data.universe import UniverseManager
        uni_cfg = cfg.get("universe", {})
        if uni_cfg.get("source") == "csv" and "csv_path" in uni_cfg and "path" not in uni_cfg:
            uni_cfg["path"] = uni_cfg["csv_path"]
        tickers = UniverseManager.load(cfg)
    except Exception as exc:
        st.error(f"Failed to load universe: {exc}")
        return

    # ── 2. Fetch prices ──────────────────────────────────────────────
    progress.progress(15, text=f"Fetching prices for {len(tickers)} tickers…")
    try:
        if store.has_cache("prices"):
            prices_df = store.load_prices("prices")
        else:
            from src.data.yahoo_fetcher import YahooFinanceSource
            source = YahooFinanceSource()
            dates_cfg = cfg.get("dates", {})
            prices_df = source.fetch_prices(
                tickers, str(dates_cfg["start"]), str(dates_cfg["end"]),
            )
            store.save_prices(prices_df, "prices")
    except Exception as exc:
        st.error(f"Price fetch failed: {exc}")
        return

    # Ensure benchmark in data
    bench_ticker = cfg.get("benchmark", {}).get("ticker", "SPY")
    if bench_ticker not in prices_df["ticker"].unique():
        try:
            from src.data.yahoo_fetcher import YahooFinanceSource
            source = YahooFinanceSource()
            dates_cfg = cfg.get("dates", {})
            bench_df = source.fetch_prices(
                [bench_ticker], str(dates_cfg["start"]), str(dates_cfg["end"]),
            )
            prices_df = pd.concat([prices_df, bench_df], ignore_index=True)
        except Exception:
            logger.warning("Could not fetch benchmark %s", bench_ticker)

    st.session_state["prices_df"] = prices_df

    # ── 3. Filter universe ───────────────────────────────────────────
    progress.progress(25, text="Filtering universe by history…")
    try:
        from src.data.universe import UniverseManager
        min_days = cfg.get("universe", {}).get("min_history_days", 252)
        tickers = UniverseManager.filter_by_history(tickers, prices_df, min_days)
        prices_df = prices_df[
            prices_df["ticker"].isin(set(tickers) | {bench_ticker})
        ].copy()
    except Exception:
        logger.warning("Universe filtering skipped")

    # ── 4. Compute factors ───────────────────────────────────────────
    progress.progress(40, text="Computing factors…")
    try:
        registry = FactorRegistry.build_default_registry(cfg)
        factor_df = registry.compute_all(prices_df, cfg)
        store.save_factors(factor_df, "factors")
        st.session_state["factor_df"] = factor_df
    except Exception as exc:
        st.error(f"Factor computation failed: {exc}")
        return

    # ── 5. Portfolio construction & rebalancing ──────────────────────
    progress.progress(60, text="Constructing portfolio…")
    try:
        constructor = PortfolioConstructor(cfg)
        reb_engine = RebalanceEngine(constructor)
        rebalance_history = reb_engine.run(factor_df, prices_df, cfg)
    except Exception as exc:
        st.error(f"Portfolio construction failed: {exc}")
        return

    # ── 6. Backtest ──────────────────────────────────────────────────
    progress.progress(75, text="Running backtest…")
    try:
        bt_engine = BacktestEngine(cfg)
        bt_result = bt_engine.run(rebalance_history, prices_df)
        st.session_state["backtest_result"] = bt_result
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        return

    # ── 7. Benchmark equity ──────────────────────────────────────────
    progress.progress(85, text="Computing benchmark…")
    dates_cfg = cfg.get("dates", {})
    tracker = BenchmarkTracker(bench_ticker)
    benchmark_equity = tracker.compute(
        prices_df, str(dates_cfg["start"]), str(dates_cfg["end"]),
    )
    st.session_state["benchmark_equity"] = benchmark_equity

    # ── 8. Performance stats ─────────────────────────────────────────
    progress.progress(95, text="Computing statistics…")
    stats = PerformanceAnalyzer.compute_stats(
        equity_curve=bt_result.equity_curve,
        daily_returns=bt_result.daily_returns,
        benchmark_equity=benchmark_equity,
        turnover=bt_result.turnover,
    )
    st.session_state["stats"] = stats
    st.session_state["pipeline_ran"] = True

    progress.progress(100, text="Pipeline complete!")
    time.sleep(0.5)
    progress.empty()
    st.toast("Pipeline finished successfully!", icon="🎉")
    st.rerun()


# =====================================================================
# Auto-load cached data on first run
# =====================================================================

def _auto_load() -> None:
    """Silently attempt to load cached data on first visit."""
    if st.session_state.get("pipeline_ran"):
        return

    config_path = st.session_state["config_path"]
    try:
        cfg = _load_config_cached(config_path)
        st.session_state["cfg"] = cfg
    except FileNotFoundError:
        return

    store = DataStore()
    if not store.has_cache("prices") or not store.has_cache("factors"):
        return

    with st.spinner("Loading cached data and reconstructing backtest — this may take a moment..."):
        try:
            prices_df = store.load_prices("prices")
            prices_df = _merge_benchmark(prices_df, store, cfg)
            factor_df = store.load_factors("factors")
            st.session_state["prices_df"] = prices_df
            st.session_state["factor_df"] = factor_df
            _run_backtest_from_cached(cfg, prices_df, factor_df)
            st.session_state["pipeline_ran"] = True
            logger.info("Auto-loaded cached data successfully")
        except Exception:
            logger.debug("Auto-load from cache failed — user will need to run pipeline")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    _auto_load()
    _render_sidebar()
    theme = st.session_state["theme_name"]

    page_labels = [label for label, _, _ in _PAGES]
    selected = st.radio(
        "Navigation",
        options=page_labels,
        horizontal=True,
        key="nav_page",
        label_visibility="collapsed",
    )

    page_module = _PAGE_LOOKUP.get(selected, overview)
    try:
        page_module.render(theme)
    except Exception:
        logger.exception("Page render failed for %s", page_module.__name__)
        st.error(
            "This page encountered an error. "
            "Check the terminal for details."
        )


main()
