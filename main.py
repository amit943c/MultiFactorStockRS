"""Multi-Factor Stock Ranking System — main pipeline CLI.

Orchestrates the full workflow: data loading, factor computation,
portfolio construction, backtesting, charting, and reporting.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.report import ReportGenerator
from src.analytics.research import CalendarAnalyzer, DrawdownAnalyzer, RegimeAnalyzer
from src.analytics.validation import LookaheadValidator
from src.analytics.factor_analytics import FactorAnalytics
from src.backtest.benchmark import BenchmarkTracker
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceAnalyzer
from src.data.store import DataStore
from src.data.universe import UniverseManager
from src.data.yahoo_fetcher import YahooFinanceSource
from src.factors.registry import FactorRegistry
from src.portfolio.construction import PortfolioConstructor
from src.portfolio.rebalance import RebalanceEngine
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.visualization.charts import ChartFactory
from src.visualization.exporters import ChartExporter

logger = logging.getLogger(__name__)


# ── CLI argument parsing ─────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Factor Stock Ranking System — backtest pipeline",
    )
    parser.add_argument(
        "--config",
        default="config/default_config.yaml",
        help="Path to the YAML config file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data download and use cached prices from DataStore",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after the pipeline completes",
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark"],
        default=None,
        help="Chart theme override (default: read from config)",
    )
    return parser.parse_args(argv)


# ── Pipeline steps ───────────────────────────────────────────────────

def _load_universe(cfg: dict[str, Any]) -> list[str]:
    """Load the ticker universe, mapping config keys as needed."""
    uni_cfg = cfg.get("universe", {})
    if uni_cfg.get("source") == "csv" and "csv_path" in uni_cfg and "path" not in uni_cfg:
        uni_cfg["path"] = uni_cfg["csv_path"]
    return UniverseManager.load(cfg)


def _fetch_prices(
    tickers: list[str],
    cfg: dict[str, Any],
    store: DataStore,
    *,
    skip_fetch: bool,
) -> pd.DataFrame:
    """Fetch (or load from cache) price data for all tickers."""
    if skip_fetch and store.has_cache("prices"):
        logger.info("Loading cached prices (--skip-fetch)")
        return store.load_prices("prices")

    source = YahooFinanceSource()
    dates_cfg = cfg.get("dates", {})
    start, end = str(dates_cfg["start"]), str(dates_cfg["end"])

    failed: list[str] = []
    frames: list[pd.DataFrame] = []

    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            df = source.fetch_prices(batch, start, end)
            if not df.empty:
                frames.append(df)
        except Exception:
            logger.exception("Price fetch failed for batch starting at %s", batch[0])
            failed.extend(batch)

    if failed:
        logger.warning("%d ticker(s) failed during download: %s", len(failed), failed[:20])

    if not frames:
        raise RuntimeError("No price data could be retrieved — cannot continue")

    prices_df = pd.concat(frames, ignore_index=True)
    store.save_prices(prices_df, "prices")
    logger.info(
        "Price data: %d rows, %d tickers, %s → %s",
        len(prices_df),
        prices_df["ticker"].nunique(),
        prices_df["date"].min(),
        prices_df["date"].max(),
    )
    return prices_df


def _fetch_benchmark(
    prices_df: pd.DataFrame,
    cfg: dict[str, Any],
    store: DataStore,
    *,
    skip_fetch: bool,
) -> pd.DataFrame:
    """Ensure the benchmark ticker is present in the price data.

    If missing, download it separately and merge into the main panel.
    """
    bench_ticker = cfg.get("benchmark", {}).get("ticker", "SPY")
    if bench_ticker in prices_df["ticker"].unique():
        logger.info("Benchmark %s already in price data", bench_ticker)
        return prices_df

    if skip_fetch and store.has_cache("benchmark_prices"):
        bench_df = store.load_prices("benchmark_prices")
    else:
        logger.info("Fetching benchmark %s separately", bench_ticker)
        source = YahooFinanceSource()
        dates_cfg = cfg.get("dates", {})
        try:
            bench_df = source.fetch_prices(
                [bench_ticker],
                str(dates_cfg["start"]),
                str(dates_cfg["end"]),
            )
            store.save_prices(bench_df, "benchmark_prices")
        except Exception:
            logger.exception("Failed to fetch benchmark %s", bench_ticker)
            return prices_df

    if not bench_df.empty:
        prices_df = pd.concat([prices_df, bench_df], ignore_index=True)
        logger.info("Merged benchmark data — total rows: %d", len(prices_df))
    return prices_df


def _generate_charts(
    bt_result: Any,
    benchmark_equity: pd.Series,
    factor_df: pd.DataFrame,
    cfg: dict[str, Any],
    theme: str,
) -> dict[str, Any]:
    """Build every chart the pipeline supports and return a name→figure dict."""
    charts: dict[str, Any] = {}

    charts["equity_curve"] = ChartFactory.equity_curve(
        bt_result.equity_curve, benchmark_equity, theme=theme,
    )
    charts["rolling_drawdown"] = ChartFactory.rolling_drawdown(
        bt_result.equity_curve, theme=theme,
    )
    charts["rolling_sharpe_chart"] = ChartFactory.rolling_sharpe_chart(
        bt_result.daily_returns, theme=theme,
    )
    charts["monthly_returns_heatmap"] = ChartFactory.monthly_returns_heatmap(
        bt_result.daily_returns, theme=theme,
    )

    factor_cols = [
        c for c in factor_df.columns
        if c not in {"date", "ticker", "composite_score", "composite_rank"}
    ]
    if factor_cols:
        charts["factor_correlation_heatmap"] = ChartFactory.factor_correlation_heatmap(
            factor_df, factor_cols, theme=theme,
        )

    if len(bt_result.turnover) > 0:
        charts["turnover_over_time"] = ChartFactory.turnover_over_time(
            bt_result.turnover, theme=theme,
        )

    last_date = _last_factor_date(factor_df)
    if last_date is not None:
        charts["top_ranked_table"] = ChartFactory.top_ranked_table(
            factor_df, last_date, theme=theme,
        )
        charts["factor_score_distribution"] = ChartFactory.factor_score_distribution(
            factor_df, last_date, theme=theme,
        )

    # ── Research diagnostics charts ──────────────────────────────────
    try:
        ic_series = FactorAnalytics.rolling_ic(
            factor_df, "composite_score", "return_1m", window=12,
        )
        if len(ic_series.dropna()) > 0:
            charts["rolling_ic"] = ChartFactory.rolling_ic_chart(
                ic_series, "composite_score", theme=theme,
            )
    except Exception:
        logger.debug("Rolling IC chart skipped")

    try:
        quantile_df = FactorAnalytics.quantile_cumulative_returns(
            factor_df, "composite_score", "return_1m", n_quantiles=5,
        )
        if not quantile_df.empty:
            charts["quantile_returns"] = ChartFactory.quantile_returns_chart(
                quantile_df, theme=theme,
            )
    except Exception:
        logger.debug("Quantile returns chart skipped")

    try:
        spread = FactorAnalytics.long_short_spread(
            factor_df, "composite_score", "return_1m",
        )
        if len(spread.dropna()) > 0:
            charts["long_short_spread"] = ChartFactory.long_short_spread_chart(
                spread, theme=theme,
            )
    except Exception:
        logger.debug("Long-short spread chart skipped")

    try:
        overlap = FactorAnalytics.holdings_overlap(bt_result.holdings_history)
        if len(overlap) > 0:
            charts["holdings_overlap"] = ChartFactory.holdings_overlap_chart(
                overlap, theme=theme,
            )
    except Exception:
        logger.debug("Holdings overlap chart skipped")

    try:
        dispersion = FactorAnalytics.ranking_dispersion(factor_df)
        if len(dispersion.dropna()) > 0:
            charts["score_dispersion"] = ChartFactory.score_dispersion_chart(
                dispersion, theme=theme,
            )
    except Exception:
        logger.debug("Score dispersion chart skipped")

    try:
        yearly = CalendarAnalyzer.yearly_returns(bt_result.daily_returns)
        if len(yearly) > 0:
            charts["yearly_returns"] = ChartFactory.yearly_returns_bar(
                yearly, theme=theme,
            )
    except Exception:
        logger.debug("Yearly returns chart skipped")

    try:
        charts["before_after_costs"] = ChartFactory.before_after_costs_chart(
            bt_result.equity_curve_gross, bt_result.equity_curve, theme=theme,
        )
    except Exception:
        logger.debug("Before/after costs chart skipped")

    try:
        episodes = DrawdownAnalyzer.worst_episodes(bt_result.equity_curve, top_n=5)
        if len(episodes) > 0:
            charts["drawdown_episodes"] = ChartFactory.drawdown_episodes_table(
                episodes, theme=theme,
            )
    except Exception:
        logger.debug("Drawdown episodes table skipped")

    return charts


def _write_research_diagnostics(
    validation_report: Any,
    factor_df: pd.DataFrame,
    bt_result: Any,
    stats: dict[str, float],
    cfg: dict[str, Any],
    output_path: str,
) -> None:
    """Write a combined research diagnostics report to markdown."""
    lines: list[str] = []
    _w = lines.append

    _w("# Research Diagnostics Report\n")
    _w(f"*Generated {__import__('datetime').datetime.now(__import__('datetime').timezone.utc):%Y-%m-%d %H:%M UTC}*\n")

    _w(validation_report.to_markdown())
    _w("")

    factor_cols = [
        c for c in factor_df.columns
        if c not in {"date", "ticker", "composite_score", "composite_rank"}
    ]

    _w("## Factor IC Summary\n")
    try:
        non_return_cols = [c for c in factor_cols if not c.startswith("return_")]
        fwd_col = "return_1m" if "return_1m" in factor_df.columns else None
        if non_return_cols and fwd_col:
            ic_summary = FactorAnalytics.factor_ic_summary(
                factor_df.dropna(subset=[fwd_col]),
                non_return_cols,
                forward_return_col=fwd_col,
            )
            _w(ic_summary.to_markdown())
        else:
            _w("*Insufficient data for IC summary (need non-return factors and a forward return column).*")
    except Exception:
        _w("*IC summary computation failed.*")
    _w("")

    _w("## Score Persistence\n")
    try:
        persistence = FactorAnalytics.score_persistence(factor_df)
        avg_p = persistence.mean()
        _w(f"- Average score rank autocorrelation: **{avg_p:.4f}**")
        _w(f"- Min: {persistence.min():.4f}, Max: {persistence.max():.4f}")
    except Exception:
        _w("*Persistence computation failed.*")
    _w("")

    _w("## Holdings Overlap\n")
    try:
        overlap = FactorAnalytics.holdings_overlap(bt_result.holdings_history)
        _w(f"- Average Jaccard overlap: **{overlap.mean():.4f}**")
        _w(f"- Min: {overlap.min():.4f}, Max: {overlap.max():.4f}")
    except Exception:
        _w("*Overlap computation failed.*")
    _w("")

    _w("## Drawdown Episodes\n")
    try:
        episodes = DrawdownAnalyzer.worst_episodes(bt_result.equity_curve, top_n=5)
        if len(episodes) > 0:
            _w(episodes.to_markdown(index=False))
        else:
            _w("*No significant drawdown episodes.*")
    except Exception:
        _w("*Drawdown analysis failed.*")
    _w("")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")


def _last_factor_date(factor_df: pd.DataFrame):
    """Return the most recent date in the factor DataFrame."""
    if "date" in factor_df.columns and len(factor_df) > 0:
        return factor_df["date"].max()
    return None


def _print_summary(stats: dict[str, float], output_cfg: dict[str, Any]) -> None:
    """Print a formatted summary to the console."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  MULTI-FACTOR STRATEGY — BACKTEST SUMMARY")
    print(sep)

    fmt_table = PerformanceAnalyzer.format_stats(stats)
    for _, row in fmt_table.iterrows():
        print(f"  {row['Metric']:<30s}  {row['Value']:>12s}")

    print(sep)
    print(f"\n  Outputs saved to:")
    print(f"    Charts  : {output_cfg.get('charts_dir', 'outputs/charts')}")
    print(f"    Reports : {output_cfg.get('reports_dir', 'outputs/reports')}")
    print(f"    Tables  : {output_cfg.get('tables_dir', 'outputs/tables')}")
    print(f"{sep}\n")


# ── Main pipeline ────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """Run the full multi-factor ranking pipeline."""
    t0 = time.perf_counter()
    args = _parse_args(argv)

    # ── 1. Load config ───────────────────────────────────────────────
    cfg = load_config(args.config)
    setup_logging(cfg)
    logger.info("Pipeline starting — config: %s", args.config)

    theme = args.theme or cfg.get("output", {}).get("theme", "light")
    output_cfg: dict[str, Any] = cfg.get("output", {})
    store = DataStore()

    # ── 2. Load universe ─────────────────────────────────────────────
    logger.info("Step 1/9  Loading ticker universe")
    tickers = _load_universe(cfg)
    logger.info("Universe: %d tickers", len(tickers))

    # ── 3. Fetch price data ──────────────────────────────────────────
    logger.info("Step 2/9  Fetching price data")
    prices_df = _fetch_prices(tickers, cfg, store, skip_fetch=args.skip_fetch)

    # ── 4. Fetch benchmark data ──────────────────────────────────────
    logger.info("Step 3/9  Ensuring benchmark data")
    prices_df = _fetch_benchmark(prices_df, cfg, store, skip_fetch=args.skip_fetch)

    # ── 5. Filter universe by history ────────────────────────────────
    min_days = cfg.get("universe", {}).get("min_history_days", 252)
    tickers = UniverseManager.filter_by_history(tickers, prices_df, min_days)
    logger.info("Post-filter universe: %d tickers", len(tickers))

    if not tickers:
        logger.error("No tickers survived the history filter — aborting")
        sys.exit(1)

    prices_df = prices_df[
        prices_df["ticker"].isin(
            set(tickers) | {cfg.get("benchmark", {}).get("ticker", "SPY")}
        )
    ].copy()

    # ── 6. Compute factors ───────────────────────────────────────────
    logger.info("Step 4/9  Computing factors")
    registry = FactorRegistry.build_default_registry(cfg)
    factor_df = registry.compute_all(prices_df, cfg)
    store.save_factors(factor_df, "factors")
    logger.info("Factor DataFrame: %d rows × %d cols", *factor_df.shape)

    # ── 7. Portfolio construction & rebalancing ──────────────────────
    logger.info("Step 5/9  Portfolio construction & rebalancing")
    constructor = PortfolioConstructor(cfg)
    engine = RebalanceEngine(constructor)
    rebalance_history = engine.run(factor_df, prices_df, cfg)
    logger.info("Rebalances: %d dates", len(rebalance_history.rebalance_dates))

    # ── 8. Backtest ──────────────────────────────────────────────────
    logger.info("Step 6/9  Running backtest")
    bt_engine = BacktestEngine(cfg)
    bt_result = bt_engine.run(rebalance_history, prices_df)
    logger.info(
        "Backtest: %d days, final equity=%.4f",
        len(bt_result.equity_curve),
        bt_result.equity_curve.iloc[-1] if len(bt_result.equity_curve) else 0.0,
    )

    # ── 9. Benchmark equity ──────────────────────────────────────────
    logger.info("Step 7/9  Computing benchmark equity curve")
    bench_ticker = cfg.get("benchmark", {}).get("ticker", "SPY")
    dates_cfg = cfg.get("dates", {})
    tracker = BenchmarkTracker(bench_ticker)
    benchmark_equity = tracker.compute(
        prices_df, str(dates_cfg["start"]), str(dates_cfg["end"]),
    )

    # ── 10. Performance stats ────────────────────────────────────────
    logger.info("Step 8/9  Computing performance statistics")
    stats = PerformanceAnalyzer.compute_stats(
        equity_curve=bt_result.equity_curve,
        daily_returns=bt_result.daily_returns,
        benchmark_equity=benchmark_equity,
        turnover=bt_result.turnover,
    )

    # ── 11. Validation & research diagnostics ────────────────────────
    logger.info("Step 9/11  Running validation and research diagnostics")
    try:
        validation_report = LookaheadValidator.run_all(
            factor_df, prices_df, rebalance_history, bt_result,
        )
        val_dir = output_cfg.get("reports_dir", "outputs/reports")
        val_path = str(Path(val_dir) / "research_diagnostics_report.md")
        _write_research_diagnostics(
            validation_report, factor_df, bt_result, stats, cfg, val_path,
        )
        logger.info("Research diagnostics report written to %s", val_path)
    except Exception:
        logger.exception("Validation / research diagnostics failed — continuing")

    # ── 12. Charts & report ──────────────────────────────────────────
    logger.info("Step 10/11  Generating charts and report")
    charts = _generate_charts(bt_result, benchmark_equity, factor_df, cfg, theme)

    chart_dir = output_cfg.get("charts_dir", "outputs/charts")
    chart_fmt = output_cfg.get("chart_format", "png")
    try:
        ChartExporter.save_all_charts(charts, chart_dir, format=chart_fmt)
        logger.info("Saved %d chart(s) to %s", len(charts), chart_dir)
    except Exception:
        logger.exception("Chart export failed — continuing without charts")

    report_dir = output_cfg.get("reports_dir", "outputs/reports")
    report_path = str(Path(report_dir) / "backtest_report.md")
    try:
        ReportGenerator.generate_markdown(stats, bt_result, cfg, report_path)
        logger.info("Report written to %s", report_path)
    except Exception:
        logger.exception("Report generation failed — continuing")

    # ── 13. Console summary ──────────────────────────────────────────
    logger.info("Step 11/11  Done")
    _print_summary(stats, output_cfg)

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline finished in %.1f seconds", elapsed)

    # ── 13. Optionally launch dashboard ──────────────────────────────
    if args.dashboard:
        _launch_dashboard()


def _launch_dashboard() -> None:
    """Spawn a Streamlit dashboard in a subprocess."""
    dashboard_path = Path("app/dashboard.py")
    if not dashboard_path.exists():
        logger.warning("Dashboard file %s not found — skipping", dashboard_path)
        return
    logger.info("Launching Streamlit dashboard…")
    subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
    )


if __name__ == "__main__":
    main()
