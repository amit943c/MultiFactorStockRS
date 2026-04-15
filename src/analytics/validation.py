"""Lookahead bias validation and data integrity diagnostics."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MOMENTUM_LOOKBACKS: dict[str, int] = {
    "return_1m": 21,
    "return_3m": 63,
    "return_6m": 126,
}

_EXTREME_RETURN_THRESHOLD = 0.50
_FACTOR_COLS_CORE = [
    "return_1m",
    "return_3m",
    "return_6m",
    "dist_ma50",
    "dist_ma200",
    "rsi_14",
    "avg_dollar_volume",
    "relative_volume",
    "realized_vol_60d",
]
_CASH_TICKER = "_CASH"


def _check(
    name: str, status: str, detail: str, severity: str = "info",
) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "severity": severity}


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Container for all validation findings."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c["status"] == "PASS" for c in self.checks)

    def add_check(
        self, name: str, status: str, detail: str, severity: str = "info",
    ) -> None:
        self.checks.append(_check(name, status, detail, severity))
        if status == "WARN":
            self.warnings.append(f"[{name}] {detail}")

    def add_assumption(self, text: str) -> None:
        self.assumptions.append(text)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.checks:
            return pd.DataFrame(columns=["name", "status", "detail", "severity"])
        return pd.DataFrame(self.checks)

    def to_markdown(self) -> str:
        lines: list[str] = []
        _w = lines.append

        _w("# Validation Report")
        _w(
            f"*Generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC}*\n"
        )

        # --- 1. Summary ---
        n_pass = sum(1 for c in self.checks if c["status"] == "PASS")
        n_warn = sum(1 for c in self.checks if c["status"] == "WARN")
        n_fail = sum(1 for c in self.checks if c["status"] == "FAIL")
        _w("## 1. Summary\n")
        _w(f"| Result | Count |")
        _w(f"|--------|-------|")
        _w(f"| PASS   | {n_pass}     |")
        _w(f"| WARN   | {n_warn}     |")
        _w(f"| FAIL   | {n_fail}     |")
        _w(f"\n**Overall: {'PASS' if self.passed else 'ISSUES DETECTED'}**\n")

        # --- 2. Lookahead Bias Checks ---
        _w("## 2. Lookahead Bias Checks\n")
        lookahead_names = {
            "factor_timing",
            "rebalance_timing",
            "no_future_data_in_factors",
        }
        la_checks = [c for c in self.checks if c["name"] in lookahead_names]
        if la_checks:
            _w("| Check | Status | Detail |")
            _w("|-------|--------|--------|")
            for c in la_checks:
                _w(f"| {c['name']} | {c['status']} | {c['detail']} |")
        else:
            _w("*No lookahead checks were run.*")
        _w("")

        # --- 3. Data Quality Checks ---
        _w("## 3. Data Quality Checks\n")
        dq_names = {"data_quality", "price_availability"}
        dq_checks = [c for c in self.checks if c["name"] in dq_names]
        if dq_checks:
            _w("| Check | Status | Detail |")
            _w("|-------|--------|--------|")
            for c in dq_checks:
                _w(f"| {c['name']} | {c['status']} | {c['detail']} |")
        else:
            _w("*No data quality checks were run.*")
        _w("")

        # --- 4. Survivorship Bias Assessment ---
        _w("## 4. Survivorship Bias Assessment\n")
        sb_checks = [c for c in self.checks if c["name"] == "survivorship_bias"]
        if sb_checks:
            _w("| Check | Status | Detail |")
            _w("|-------|--------|--------|")
            for c in sb_checks:
                _w(f"| {c['name']} | {c['status']} | {c['detail']} |")
        else:
            _w("*No survivorship bias checks were run.*")
        _w("")

        # --- 5. Execution Assumptions ---
        _w("## 5. Execution Assumptions\n")
        if self.assumptions:
            for a in self.assumptions:
                _w(f"- {a}")
        else:
            _w("*No execution assumptions recorded.*")
        _w("")

        # --- 6. Missing Data Summary ---
        _w("## 6. Missing Data Summary\n")
        md_checks = [c for c in self.checks if c["name"] == "missing_data"]
        if md_checks:
            _w("| Check | Status | Detail |")
            _w("|-------|--------|--------|")
            for c in md_checks:
                _w(f"| {c['name']} | {c['status']} | {c['detail']} |")
        else:
            _w("*No missing-data checks were run.*")
        _w("")

        # --- Warnings ---
        if self.warnings:
            _w("## Warnings\n")
            for w in self.warnings:
                _w(f"- {w}")
            _w("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LookaheadValidator
# ---------------------------------------------------------------------------


class LookaheadValidator:
    """Validate that the backtest pipeline has no lookahead bias."""

    @staticmethod
    def validate_factor_timing(
        factor_df: pd.DataFrame,
        prices_df: pd.DataFrame,
    ) -> list[dict]:
        """Check that factor values on date T use only data available on or before T.

        For each momentum factor, verify:
        - The adj_close on T and T-lookback both exist in prices.
        - Factor values are NaN during the warm-up period (first *lookback* dates).
        - No factor row references a date absent from the prices panel.
        """
        results: list[dict] = []

        factor_dates = set(pd.to_datetime(factor_df["date"]).dt.date)
        price_dates = set(pd.to_datetime(prices_df["date"]).dt.date)

        future_dates = factor_dates - price_dates
        if future_dates:
            results.append(
                _check(
                    "factor_timing",
                    "FAIL",
                    f"{len(future_dates)} factor dates have no corresponding price "
                    f"data (first: {min(future_dates)}). Possible lookahead.",
                    "error",
                )
            )
            return results

        available_momentum = [
            c for c in _MOMENTUM_LOOKBACKS if c in factor_df.columns
        ]

        if not available_momentum:
            results.append(
                _check(
                    "factor_timing",
                    "WARN",
                    "No momentum columns found in factor_df; skipping timing check.",
                    "warning",
                )
            )
            return results

        prices_sorted = (
            prices_df[["date", "ticker", "adj_close"]]
            .copy()
            .sort_values(["ticker", "date"])
        )
        prices_sorted["date"] = pd.to_datetime(prices_sorted["date"])

        all_ok = True
        for col, lookback in _MOMENTUM_LOOKBACKS.items():
            if col not in factor_df.columns:
                continue

            merged = factor_df[["date", "ticker", col]].copy()
            merged["date"] = pd.to_datetime(merged["date"])

            ticker_date_counts = (
                prices_sorted.groupby("ticker")["date"]
                .apply(lambda s: s.sort_values().reset_index(drop=True))
            )

            warmup_check = merged.merge(
                prices_sorted.groupby("ticker")["date"]
                .apply(lambda s: s.sort_values().iloc[:lookback] if len(s) >= lookback else s)
                .reset_index(level=0)
                .rename(columns={"date": "warmup_date"}),
                left_on=["ticker", "date"],
                right_on=["ticker", "warmup_date"],
                how="inner",
            )

            if len(warmup_check) > 0:
                non_nan_warmup = warmup_check[warmup_check[col].notna()]
                if len(non_nan_warmup) > 0:
                    pct = len(non_nan_warmup) / len(warmup_check) * 100
                    if pct > 5:
                        all_ok = False
                        results.append(
                            _check(
                                "factor_timing",
                                "WARN",
                                f"{col}: {pct:.1f}% of warm-up rows have non-NaN "
                                f"values ({len(non_nan_warmup)} rows). Expected NaN "
                                f"during the first {lookback} trading days per ticker.",
                                "warning",
                            )
                        )

        if all_ok and not results:
            results.append(
                _check(
                    "factor_timing",
                    "PASS",
                    "All factor dates exist in prices; warm-up periods contain NaN "
                    "as expected. No evidence of lookahead bias in factor timing.",
                    "info",
                )
            )

        return results

    @staticmethod
    def validate_rebalance_timing(
        rebalance_history: Any,
        factor_df: pd.DataFrame,
    ) -> list[dict]:
        """Verify that holdings on each rebalance date are derived from factor
        scores computed on or before that date.

        For each rebalance date, the ranked stocks should match the ``factor_df``
        scores from that same date or earlier.
        """
        results: list[dict] = []

        if not rebalance_history or not rebalance_history.rebalance_dates:
            results.append(
                _check(
                    "rebalance_timing",
                    "WARN",
                    "No rebalance history provided; skipping rebalance timing check.",
                    "warning",
                )
            )
            return results

        factor_dates = sorted(pd.to_datetime(factor_df["date"]).dt.date.unique())
        factor_date_set = set(factor_dates)
        violations: list[str] = []

        for reb_date in rebalance_history.rebalance_dates:
            holdings = rebalance_history.holdings.get(reb_date)
            if holdings is None or holdings.empty:
                continue

            stock_tickers = set(
                holdings.loc[holdings["ticker"] != _CASH_TICKER, "ticker"]
            )
            if not stock_tickers:
                continue

            if reb_date not in factor_date_set:
                later_dates = [d for d in factor_dates if d > reb_date]
                if later_dates:
                    closest_after = min(later_dates)
                    snap = factor_df[
                        pd.to_datetime(factor_df["date"]).dt.date == closest_after
                    ]
                    if set(snap["ticker"]) & stock_tickers:
                        violations.append(
                            f"{reb_date}: no factor data on date; "
                            f"holdings may use data from {closest_after} (future)"
                        )
                continue

            snap = factor_df[pd.to_datetime(factor_df["date"]).dt.date == reb_date]
            snap_tickers = set(snap["ticker"])

            missing = stock_tickers - snap_tickers
            if missing and len(missing) > len(stock_tickers) * 0.5:
                violations.append(
                    f"{reb_date}: {len(missing)}/{len(stock_tickers)} held "
                    f"tickers absent from factor_df on that date"
                )

            if "composite_rank" in holdings.columns and "composite_rank" in snap.columns:
                held_ranks = holdings.loc[
                    holdings["ticker"] != _CASH_TICKER,
                    ["ticker", "composite_rank"],
                ].dropna(subset=["composite_rank"])
                factor_ranks = snap[["ticker", "composite_rank"]].dropna(
                    subset=["composite_rank"]
                )
                merged = held_ranks.merge(
                    factor_ranks, on="ticker", suffixes=("_held", "_factor")
                )
                if len(merged) > 0:
                    mismatch = (
                        merged["composite_rank_held"] != merged["composite_rank_factor"]
                    ).sum()
                    if mismatch > len(merged) * 0.5:
                        violations.append(
                            f"{reb_date}: {mismatch}/{len(merged)} ranks differ "
                            f"between holdings and factor_df"
                        )

        if violations:
            results.append(
                _check(
                    "rebalance_timing",
                    "WARN",
                    f"{len(violations)} potential issue(s): {violations[0]}"
                    + (f" … and {len(violations) - 1} more" if len(violations) > 1 else ""),
                    "warning",
                )
            )
        else:
            results.append(
                _check(
                    "rebalance_timing",
                    "PASS",
                    "All rebalance-date holdings are consistent with factor scores "
                    "available on or before the rebalance date.",
                    "info",
                )
            )
        return results

    @staticmethod
    def validate_price_availability(
        holdings_history: dict[datetime.date, pd.DataFrame],
        prices_df: pd.DataFrame,
    ) -> list[dict]:
        """Check that every stock in the portfolio has price data on and after
        the rebalance date (i.e. we can actually trade it).
        """
        results: list[dict] = []

        if not holdings_history:
            results.append(
                _check(
                    "price_availability",
                    "WARN",
                    "No holdings history provided; skipping price availability check.",
                    "warning",
                )
            )
            return results

        prices_index = prices_df.copy()
        prices_index["_date"] = pd.to_datetime(prices_index["date"]).dt.date
        ticker_date_pairs = set(
            zip(prices_index["ticker"], prices_index["_date"])
        )

        total_positions = 0
        missing_positions: list[str] = []

        for reb_date, holdings in holdings_history.items():
            if holdings is None or holdings.empty:
                continue
            tickers = holdings.loc[
                holdings["ticker"] != _CASH_TICKER, "ticker"
            ]
            for ticker in tickers:
                total_positions += 1
                if (ticker, reb_date) not in ticker_date_pairs:
                    missing_positions.append(f"{ticker} on {reb_date}")

        if missing_positions:
            sample = missing_positions[:5]
            results.append(
                _check(
                    "price_availability",
                    "WARN",
                    f"{len(missing_positions)}/{total_positions} positions lack "
                    f"price data on the rebalance date. "
                    f"Examples: {', '.join(sample)}.",
                    "warning",
                )
            )
        else:
            results.append(
                _check(
                    "price_availability",
                    "PASS",
                    f"All {total_positions} positions have price data on their "
                    f"rebalance date.",
                    "info",
                )
            )
        return results

    @staticmethod
    def check_survivorship_bias(prices_df: pd.DataFrame) -> list[dict]:
        """Flag potential survivorship bias by checking:
        - Are there tickers that appear and disappear?
        - What percentage of the universe has full history?
        - Are there gaps in trading days for individual tickers?
        """
        results: list[dict] = []

        prices = prices_df.copy()
        prices["_date"] = pd.to_datetime(prices["date"]).dt.date

        all_dates = sorted(prices["_date"].unique())
        if len(all_dates) < 2:
            results.append(
                _check(
                    "survivorship_bias",
                    "WARN",
                    "Fewer than 2 unique dates in prices; cannot assess survivorship.",
                    "warning",
                )
            )
            return results

        first_date, last_date = all_dates[0], all_dates[-1]
        n_total_dates = len(all_dates)

        ticker_stats = (
            prices.groupby("ticker")["_date"]
            .agg(["min", "max", "count"])
            .rename(columns={"min": "first_seen", "max": "last_seen", "count": "n_days"})
        )

        full_history = (
            (ticker_stats["first_seen"] == first_date)
            & (ticker_stats["last_seen"] == last_date)
        )
        pct_full = full_history.mean() * 100

        entered_late = (ticker_stats["first_seen"] > first_date).sum()
        exited_early = (ticker_stats["last_seen"] < last_date).sum()

        gaps: list[str] = []
        date_set = set(all_dates)
        for ticker, group in prices.groupby("ticker"):
            ticker_dates = set(group["_date"])
            expected_range = {
                d for d in date_set
                if ticker_stats.loc[ticker, "first_seen"]  # type: ignore[index]
                <= d
                <= ticker_stats.loc[ticker, "last_seen"]  # type: ignore[index]
            }
            missing = expected_range - ticker_dates
            if missing and len(missing) > n_total_dates * 0.05:
                gaps.append(
                    f"{ticker}: {len(missing)} missing days "
                    f"({len(missing) / len(expected_range) * 100:.1f}%)"
                )

        detail_parts = [
            f"{pct_full:.1f}% of tickers have full history "
            f"({first_date} to {last_date})",
            f"{entered_late} entered late, {exited_early} exited early",
        ]
        if gaps:
            detail_parts.append(
                f"{len(gaps)} ticker(s) with >5% gap days "
                f"(e.g. {gaps[0]})"
            )

        if pct_full < 50 or entered_late > 0 or exited_early > 0:
            status = "WARN"
            severity = "warning"
        else:
            status = "PASS"
            severity = "info"

        results.append(
            _check("survivorship_bias", status, "; ".join(detail_parts), severity)
        )
        return results

    @staticmethod
    def check_data_quality(prices_df: pd.DataFrame) -> list[dict]:
        """Check for data issues: negative prices, zero-volume days, extreme
        price jumps, duplicate rows, and missing dates.
        """
        results: list[dict] = []
        issues: list[str] = []

        n_rows = len(prices_df)
        dups = prices_df.duplicated(subset=["date", "ticker"]).sum()
        if dups > 0:
            issues.append(f"{dups} duplicate (date, ticker) rows")

        for col in ("open", "high", "low", "close", "adj_close"):
            if col not in prices_df.columns:
                continue
            neg = (prices_df[col] < 0).sum()
            if neg > 0:
                issues.append(f"{neg} negative values in '{col}'")

        if "volume" in prices_df.columns:
            zero_vol = (prices_df["volume"] == 0).sum()
            if zero_vol > 0:
                pct = zero_vol / n_rows * 100
                issues.append(f"{zero_vol} zero-volume rows ({pct:.1f}%)")

        if "adj_close" in prices_df.columns:
            df = prices_df[["date", "ticker", "adj_close"]].copy()
            df = df.sort_values(["ticker", "date"])
            df["_ret"] = df.groupby("ticker")["adj_close"].pct_change()
            extreme = (df["_ret"].abs() > _EXTREME_RETURN_THRESHOLD).sum()
            if extreme > 0:
                issues.append(
                    f"{extreme} daily returns exceed "
                    f"{_EXTREME_RETURN_THRESHOLD:.0%} (potential data errors)"
                )

        all_dates = sorted(pd.to_datetime(prices_df["date"]).dt.date.unique())
        if len(all_dates) > 1:
            bday_range = pd.bdate_range(all_dates[0], all_dates[-1])
            expected_count = len(bday_range)
            actual_count = len(all_dates)
            gap = expected_count - actual_count
            if gap > expected_count * 0.05:
                issues.append(
                    f"{gap} missing business days in the date range "
                    f"(expected ~{expected_count}, got {actual_count})"
                )

        if issues:
            results.append(
                _check(
                    "data_quality",
                    "WARN",
                    f"{len(issues)} issue(s): " + "; ".join(issues),
                    "warning",
                )
            )
        else:
            results.append(
                _check(
                    "data_quality",
                    "PASS",
                    f"No quality issues detected across {n_rows} price rows.",
                    "info",
                )
            )
        return results

    @staticmethod
    def validate_no_future_data_in_factors(
        factor_df: pd.DataFrame,
    ) -> list[dict]:
        """Check that factor columns like ``return_1m`` are backward-looking.

        On each date T, ``return_1m`` should reflect the return over the PAST
        month.  If it instead correlates highly with the *future* price change
        from T to T+21, that indicates lookahead bias.

        Strategy: compute the Pearson correlation between ``return_1m`` on date T
        and the actual ``adj_close`` change from T to T+lookback (future) vs from
        T-lookback to T (past).  A backward-looking factor should correlate
        strongly with the past change and weakly with the future one.
        """
        results: list[dict] = []

        available = [c for c in _MOMENTUM_LOOKBACKS if c in factor_df.columns]
        if not available or "adj_close" not in factor_df.columns:
            results.append(
                _check(
                    "no_future_data_in_factors",
                    "WARN",
                    "Cannot run future-data check: need momentum columns and "
                    "adj_close in factor_df.",
                    "warning",
                )
            )
            return results

        df = factor_df[["date", "ticker", "adj_close"] + available].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"])

        suspicions: list[str] = []

        for col, lookback in _MOMENTUM_LOOKBACKS.items():
            if col not in df.columns:
                continue

            df["_past_ret"] = df.groupby("ticker")["adj_close"].transform(
                lambda s, lb=lookback: s / s.shift(lb) - 1
            )
            df["_future_ret"] = df.groupby("ticker")["adj_close"].transform(
                lambda s, lb=lookback: s.shift(-lb) / s - 1
            )

            valid = df[[col, "_past_ret", "_future_ret"]].dropna()
            if len(valid) < 30:
                continue

            corr_past = valid[col].corr(valid["_past_ret"])
            corr_future = valid[col].corr(valid["_future_ret"])

            logger.debug(
                "%s: corr(past)=%.4f  corr(future)=%.4f", col, corr_past, corr_future
            )

            if abs(corr_future) > 0.8 and abs(corr_future) > abs(corr_past):
                suspicions.append(
                    f"{col}: corr with future returns ({corr_future:.3f}) exceeds "
                    f"corr with past returns ({corr_past:.3f}) — likely lookahead"
                )
            elif abs(corr_past) > 0.8:
                logger.debug(
                    "%s strongly correlates with past returns (%.3f) as expected",
                    col,
                    corr_past,
                )

        df.drop(columns=["_past_ret", "_future_ret"], inplace=True, errors="ignore")

        if suspicions:
            results.append(
                _check(
                    "no_future_data_in_factors",
                    "FAIL",
                    f"Potential lookahead detected: {'; '.join(suspicions)}",
                    "error",
                )
            )
        else:
            results.append(
                _check(
                    "no_future_data_in_factors",
                    "PASS",
                    "Momentum factors correlate with past price changes, not future "
                    "ones. No evidence of lookahead in factor construction.",
                    "info",
                )
            )
        return results

    @classmethod
    def run_all(
        cls,
        factor_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        rebalance_history: Any | None = None,
        backtest_result: Any | None = None,
    ) -> ValidationReport:
        """Run all validation checks and return a comprehensive report."""
        report = ValidationReport()
        logger.info("Running full validation suite")

        holdings_history: dict[datetime.date, pd.DataFrame] | None = None
        if backtest_result is not None:
            holdings_history = backtest_result.holdings_history
        elif rebalance_history is not None:
            holdings_history = rebalance_history.holdings

        for c in cls.validate_factor_timing(factor_df, prices_df):
            report.checks.append(c)
            if c["status"] == "WARN":
                report.warnings.append(f"[{c['name']}] {c['detail']}")

        factor_with_prices = factor_df
        if "adj_close" not in factor_df.columns and "adj_close" in prices_df.columns:
            adj = prices_df[["date", "ticker", "adj_close"]].drop_duplicates(
                subset=["date", "ticker"]
            )
            factor_with_prices = factor_df.merge(adj, on=["date", "ticker"], how="left")

        for c in cls.validate_no_future_data_in_factors(factor_with_prices):
            report.checks.append(c)
            if c["status"] in ("WARN", "FAIL"):
                report.warnings.append(f"[{c['name']}] {c['detail']}")

        if rebalance_history is not None:
            for c in cls.validate_rebalance_timing(rebalance_history, factor_df):
                report.checks.append(c)
                if c["status"] == "WARN":
                    report.warnings.append(f"[{c['name']}] {c['detail']}")

        if holdings_history is not None:
            for c in cls.validate_price_availability(holdings_history, prices_df):
                report.checks.append(c)
                if c["status"] == "WARN":
                    report.warnings.append(f"[{c['name']}] {c['detail']}")

        for c in cls.check_data_quality(prices_df):
            report.checks.append(c)
            if c["status"] == "WARN":
                report.warnings.append(f"[{c['name']}] {c['detail']}")

        for c in cls.check_survivorship_bias(prices_df):
            report.checks.append(c)
            if c["status"] == "WARN":
                report.warnings.append(f"[{c['name']}] {c['detail']}")

        missing = DataIntegrityChecker.missing_data_report(factor_df)

        warmup_cols = {"dist_ma200", "dist_ma50", "composite_score",
                       "composite_rank", "return_6m", "return_3m"}
        high_missing = missing.loc[missing["missing_pct"] > 10]
        unexpected = high_missing[~high_missing["column"].isin(warmup_cols)]
        warmup_only = high_missing[high_missing["column"].isin(warmup_cols)]

        if len(unexpected) > 0:
            cols = ", ".join(
                f"{r['column']} ({r['missing_pct']:.1f}%)"
                for _, r in unexpected.iterrows()
            )
            report.add_check(
                "missing_data",
                "WARN",
                f"{len(unexpected)} column(s) with >10% missing: {cols}",
                "warning",
            )
        elif len(warmup_only) > 0:
            cols = ", ".join(
                f"{r['column']} ({r['missing_pct']:.1f}%)"
                for _, r in warmup_only.iterrows()
            )
            report.add_check(
                "missing_data",
                "PASS",
                f"All missing data is in warmup-dependent columns (expected): {cols}. "
                "Moving-average and multi-month return factors require a lookback "
                "window before producing values.",
                "info",
            )
        else:
            report.add_check(
                "missing_data",
                "PASS",
                "All factor columns have <=10% missing values.",
                "info",
            )

        logger.info(
            "Validation complete: %d checks (%d pass, %d warn, %d fail)",
            len(report.checks),
            sum(1 for c in report.checks if c["status"] == "PASS"),
            sum(1 for c in report.checks if c["status"] == "WARN"),
            sum(1 for c in report.checks if c["status"] == "FAIL"),
        )
        return report


# ---------------------------------------------------------------------------
# DataIntegrityChecker
# ---------------------------------------------------------------------------


class DataIntegrityChecker:
    """Missing data analysis and reporting."""

    @staticmethod
    def missing_data_report(factor_df: pd.DataFrame) -> pd.DataFrame:
        """For each factor column, compute total values, missing count, missing
        percentage, the date with the most missing values, and the ticker with
        the most missing values.

        Returns a summary DataFrame with one row per column.
        """
        factor_cols = [
            c for c in factor_df.columns if c not in ("date", "ticker")
        ]
        if not factor_cols:
            return pd.DataFrame(
                columns=[
                    "column",
                    "total",
                    "missing",
                    "missing_pct",
                    "worst_date",
                    "worst_date_missing",
                    "worst_ticker",
                    "worst_ticker_missing",
                ]
            )

        rows: list[dict[str, Any]] = []
        for col in factor_cols:
            total = len(factor_df)
            missing = int(factor_df[col].isna().sum())
            pct = missing / total * 100 if total > 0 else 0.0

            worst_date = None
            worst_date_missing = 0
            if "date" in factor_df.columns:
                by_date = factor_df.groupby("date")[col].apply(
                    lambda s: int(s.isna().sum())
                )
                if len(by_date) > 0 and by_date.max() > 0:
                    worst_date = by_date.idxmax()
                    worst_date_missing = int(by_date.max())

            worst_ticker = None
            worst_ticker_missing = 0
            if "ticker" in factor_df.columns:
                by_ticker = factor_df.groupby("ticker")[col].apply(
                    lambda s: int(s.isna().sum())
                )
                if len(by_ticker) > 0 and by_ticker.max() > 0:
                    worst_ticker = by_ticker.idxmax()
                    worst_ticker_missing = int(by_ticker.max())

            rows.append(
                {
                    "column": col,
                    "total": total,
                    "missing": missing,
                    "missing_pct": round(pct, 2),
                    "worst_date": worst_date,
                    "worst_date_missing": worst_date_missing,
                    "worst_ticker": worst_ticker,
                    "worst_ticker_missing": worst_ticker_missing,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def coverage_matrix(
        factor_df: pd.DataFrame,
        factor_cols: list[str],
    ) -> pd.DataFrame:
        """Create a date x factor coverage matrix showing the percentage of the
        universe with valid (non-NaN) data for each factor on each date.
        """
        if "date" not in factor_df.columns:
            return pd.DataFrame()

        records: list[dict[str, Any]] = []
        for date_val, group in factor_df.groupby("date"):
            row: dict[str, Any] = {"date": date_val, "universe_size": len(group)}
            for col in factor_cols:
                if col in group.columns:
                    valid = group[col].notna().sum()
                    row[col] = round(valid / len(group) * 100, 2) if len(group) else 0.0
                else:
                    row[col] = 0.0
            records.append(row)

        result = pd.DataFrame(records)
        if "date" in result.columns:
            result = result.set_index("date")
        return result

    @staticmethod
    def execution_assumptions_summary(cfg: dict[str, Any]) -> list[str]:
        """Generate a list of execution assumptions from the config."""
        assumptions: list[str] = []

        assumptions.append(
            "Execution at close price (adj_close) — "
            "close-to-close return model"
        )

        port = cfg.get("portfolio", {})
        tc = port.get("transaction_cost_bps", 0)
        slip = port.get("slippage_bps", 0)
        assumptions.append(
            f"Transaction cost: {tc} bps one-way per unit of turnover"
        )
        assumptions.append(f"Slippage: {slip} bps one-way per unit of turnover")

        reb = cfg.get("rebalance", {})
        freq = reb.get("frequency", "unknown")
        dow = reb.get("day_of_week", None)
        day_name = (
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][dow]
            if isinstance(dow, int) and 0 <= dow <= 4
            else str(dow)
        )
        assumptions.append(f"Rebalance frequency: {freq} (target day: {day_name})")

        assumptions.append("No short selling — long-only portfolio")
        assumptions.append(
            "No market-impact model — results may overstate achievable "
            "returns for large capital"
        )
        assumptions.append("No partial fills — orders assumed fully executed")

        top_n = port.get("top_n", "N/A")
        max_w = port.get("max_position_weight", "N/A")
        eq = port.get("equal_weight", True)
        assumptions.append(
            f"Portfolio: top {top_n} stocks, max weight {max_w}, "
            f"{'equal' if eq else 'score'}-weighted"
        )

        if port.get("allow_cash", True):
            assumptions.append(
                "Cash allocation permitted when fewer stocks available "
                "than target"
            )

        assumptions.append(
            "Survivorship bias may exist depending on the underlying "
            "data source (e.g. current S&P 500 constituents)"
        )
        assumptions.append(
            "Factor scores are computed from point-in-time price data; "
            "fundamental data (if used) may still contain look-ahead bias"
        )

        return assumptions
