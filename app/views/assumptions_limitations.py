"""Assumptions and Limitations page — validation report, data quality, lookahead checks."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.analytics.validation import (
    DataIntegrityChecker,
    LookaheadValidator,
    ValidationReport,
)

logger = logging.getLogger(__name__)

_METHODOLOGY_NOTES = """
### Backtest Methodology

This backtesting system uses a **multi-factor ranking model** with the following methodology:

**Factor Construction**
- Factors are computed from point-in-time price data using rolling lookback windows.
- Composite scores are formed by weighting individual factor z-scores according to the
  configuration.
- Factor values on date T use only data available on or before T (validated by the
  lookahead checks above).

**Portfolio Construction**
- Each rebalance date, stocks are ranked by composite score and the top-N are selected.
- Weights are assigned according to the configured scheme (equal-weight or score-weighted).
- A cash allocation may be used when fewer stocks meet the selection criteria.

**Execution Model**
- Trades are assumed to execute at the **adjusted close price** on the rebalance date.
- Transaction costs and slippage are modelled as a fixed basis-point charge per unit
  of turnover.
- No market-impact model is used — results may overstate achievable returns for large
  capital deployments.

**Known Limitations**
1. **Survivorship bias**: The universe is typically defined using current index
   constituents; delisted stocks are absent from the dataset.
2. **Look-ahead in fundamentals**: If fundamental data (e.g. earnings, revenue) is
   used, the as-reported dates may not perfectly align with actual release dates.
3. **Close-price execution**: Real-world execution at the close is difficult; actual
   fills will differ due to bid/ask spread and volume constraints.
4. **No partial fills**: All orders are assumed to be fully executed, which may not
   hold for illiquid names.
5. **No short selling**: The strategy is long-only; short-selling constraints are
   not modelled.
6. **Rebalance timing**: The rebalance is triggered at a fixed frequency; real-world
   timing may vary due to holidays, corporate actions, and market closures.
"""


def render(theme: str) -> None:
    """Render the Assumptions & Limitations page."""
    st.header("Assumptions & Limitations")

    cfg: dict | None = st.session_state.get("cfg")
    factor_df: pd.DataFrame | None = st.session_state.get("factor_df")
    prices_df: pd.DataFrame | None = st.session_state.get("prices_df")
    bt_result = st.session_state.get("backtest_result")

    # ── 1. Execution Assumptions ──────────────────────────────────────
    st.subheader("Execution Assumptions")
    if cfg is not None:
        try:
            assumptions = DataIntegrityChecker.execution_assumptions_summary(cfg)
            for assumption in assumptions:
                st.markdown(f"- {assumption}")
        except Exception:
            logger.exception("Execution assumptions extraction failed")
            st.warning("Could not extract execution assumptions from config.")
    else:
        st.info("Load a configuration to see execution assumptions.")

    st.markdown("---")

    # ── 2. Lookahead Bias Validation ──────────────────────────────────
    st.subheader("Lookahead Bias & Data Quality Validation")

    if factor_df is None or prices_df is None:
        st.info(
            "Validation requires factor data and price data. "
            "Run the pipeline or load cached data first."
        )
    else:
        # Auto-run validation on first load, cache in session state
        if "al_validation_report" not in st.session_state:
            with st.spinner("Running validation suite…"):
                try:
                    report = LookaheadValidator.run_all(
                        factor_df=factor_df,
                        prices_df=prices_df,
                        rebalance_history=None,
                        backtest_result=bt_result,
                    )
                    st.session_state["al_validation_report"] = report
                except Exception:
                    logger.exception("Validation suite failed")
                    st.error("Validation failed. Check the logs for details.")

        if st.button("Re-run Validation", key="al_run_validation"):
            with st.spinner("Re-running validation suite…"):
                try:
                    report = LookaheadValidator.run_all(
                        factor_df=factor_df,
                        prices_df=prices_df,
                        rebalance_history=None,
                        backtest_result=bt_result,
                    )
                    st.session_state["al_validation_report"] = report
                except Exception:
                    logger.exception("Validation suite failed")
                    st.error("Validation failed. Check the logs for details.")

        report: ValidationReport | None = st.session_state.get("al_validation_report")

        if report is not None:
            _render_validation_summary(report)

            with st.expander("Full Validation Report (Markdown)", expanded=False):
                st.markdown(report.to_markdown())

            with st.expander("Detailed Check Results"):
                checks_df = report.to_dataframe()
                if len(checks_df) > 0:
                    def _style_status(val: str) -> str:
                        if val == "PASS":
                            return "background-color: #d4edda; color: #155724;"
                        elif val == "WARN":
                            return "background-color: #fff3cd; color: #856404;"
                        elif val == "FAIL":
                            return "background-color: #f8d7da; color: #721c24;"
                        return ""

                    st.dataframe(
                        checks_df.style.map(_style_status, subset=["status"]),
                        use_container_width=True,
                    )
                else:
                    st.info("No checks were executed.")

            if report.warnings:
                with st.expander("Warnings", expanded=True):
                    for warning in report.warnings:
                        st.warning(warning)

    st.markdown("---")

    # ── 3. Data Quality Overview ──────────────────────────────────────
    st.subheader("Data Quality Overview")

    if factor_df is not None and len(factor_df) > 0:
        _render_missing_data_report(factor_df)
        _render_coverage_matrix(factor_df)
    else:
        st.info("No factor data available for data quality analysis.")

    st.markdown("---")

    # ── 4. Survivorship Bias Assessment ───────────────────────────────
    st.subheader("Survivorship Bias Assessment")

    if prices_df is not None:
        try:
            results = LookaheadValidator.check_survivorship_bias(prices_df)
            for check in results:
                if check["status"] == "PASS":
                    st.success(f"**{check['name']}**: {check['detail']}")
                elif check["status"] == "WARN":
                    st.warning(f"**{check['name']}**: {check['detail']}")
                else:
                    st.error(f"**{check['name']}**: {check['detail']}")
        except Exception:
            logger.exception("Survivorship bias check failed")
            st.warning("Could not run survivorship bias assessment.")
    else:
        st.info("Price data required for survivorship bias assessment.")

    st.markdown("---")

    # ── 5. Methodology Notes ──────────────────────────────────────────
    st.subheader("Methodology Notes")
    with st.expander("Backtest Methodology & Known Limitations", expanded=False):
        st.markdown(_METHODOLOGY_NOTES)


# ── Helpers ───────────────────────────────────────────────────────────


def _render_validation_summary(report: ValidationReport) -> None:
    """Show a compact summary of validation results."""
    n_pass = sum(1 for c in report.checks if c["status"] == "PASS")
    n_warn = sum(1 for c in report.checks if c["status"] == "WARN")
    n_fail = sum(1 for c in report.checks if c["status"] == "FAIL")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Checks", len(report.checks))
    c2.metric("Passed", n_pass)
    c3.metric("Warnings", n_warn)
    c4.metric("Failed", n_fail)

    if report.passed:
        st.success("All validation checks passed.")
    elif n_fail > 0:
        st.error(f"{n_fail} check(s) failed — review the detailed results below.")
    else:
        st.warning(f"{n_warn} warning(s) detected — review the detailed results below.")


_WARMUP_COLS = {"dist_ma200", "dist_ma50", "composite_score",
                "composite_rank", "return_6m", "return_3m"}


def _render_missing_data_report(factor_df: pd.DataFrame) -> None:
    """Render missing data summary for factor columns."""
    try:
        missing_report = DataIntegrityChecker.missing_data_report(factor_df)
        if len(missing_report) > 0:
            with st.expander("Missing Data Report", expanded=False):
                st.dataframe(
                    missing_report.style.format({
                        "total": "{:,.0f}",
                        "missing": "{:,.0f}",
                        "missing_pct": "{:.2f}%",
                        "worst_date_missing": "{:.0f}",
                        "worst_ticker_missing": "{:.0f}",
                    }).background_gradient(
                        subset=["missing_pct"], cmap="YlOrRd",
                    ),
                    use_container_width=True,
                )

            high_missing = missing_report.loc[missing_report["missing_pct"] > 10]
            unexpected = high_missing[~high_missing["column"].isin(_WARMUP_COLS)]
            warmup_only = high_missing[high_missing["column"].isin(_WARMUP_COLS)]

            if len(unexpected) > 0:
                st.warning(
                    f"{len(unexpected)} column(s) have >10% missing values: "
                    f"{', '.join(unexpected['column'].tolist())}"
                )
            elif len(warmup_only) > 0:
                st.success(
                    "All missing data is from lookback warmup periods (expected). "
                    "No unexpected data gaps."
                )
            else:
                st.success("All factor columns have <=10% missing values.")
        else:
            st.info("No factor columns to analyze.")
    except Exception:
        logger.exception("Missing data report failed")
        st.warning("Could not generate missing data report.")


def _render_coverage_matrix(factor_df: pd.DataFrame) -> None:
    """Render the date x factor coverage matrix."""
    factor_cols = [
        c for c in factor_df.columns if c not in ("date", "ticker")
    ]
    if not factor_cols:
        return

    with st.expander("Factor Coverage Matrix (% valid per date)", expanded=False):
        try:
            coverage = DataIntegrityChecker.coverage_matrix(factor_df, factor_cols)
            if len(coverage) > 0:
                display_cols = [c for c in coverage.columns if c != "universe_size"]
                if display_cols:
                    active = coverage[
                        coverage[display_cols].mean(axis=1) > 0
                    ].sort_index(ascending=False)

                    if len(active) < len(coverage):
                        st.caption(
                            f"Showing {len(active)} of {len(coverage)} dates "
                            f"(warmup dates with 0% coverage hidden). "
                            f"Most recent dates first."
                        )

                    st.dataframe(
                        active[display_cols].style.background_gradient(
                            cmap="RdYlGn", vmin=0, vmax=100,
                        ).format("{:.1f}%"),
                        use_container_width=True,
                        height=min(len(active) * 35 + 40, 500),
                    )
                else:
                    st.info("No factor coverage data available.")
            else:
                st.info("Coverage matrix is empty.")
        except Exception:
            logger.exception("Coverage matrix failed")
            st.warning("Could not generate coverage matrix.")
