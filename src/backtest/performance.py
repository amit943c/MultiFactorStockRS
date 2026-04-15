"""Performance analytics — compute risk/return statistics from backtest outputs."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252
_WEEKS_PER_YEAR = 52


class PerformanceAnalyzer:
    """Stateless calculator for portfolio performance metrics."""

    @staticmethod
    def compute_stats(
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        benchmark_equity: pd.Series | None = None,
        turnover: pd.Series | None = None,
        rf: float = 0.0,
    ) -> dict[str, float]:
        """Compute a comprehensive set of risk-adjusted return statistics.

        Parameters
        ----------
        equity_curve:
            Cumulative equity curve (starts near 1.0), date-indexed.
        daily_returns:
            Simple daily returns, date-indexed, aligned with *equity_curve*.
        benchmark_equity:
            Optional benchmark equity curve for relative metrics.
        turnover:
            Optional per-rebalance turnover series.
        rf:
            Annualised risk-free rate (decimal, e.g. 0.04 for 4 %).

        Returns
        -------
        dict[str, float]
            Named metrics.
        """
        stats: dict[str, float] = {}

        n_days = len(daily_returns)
        if n_days < 2:
            logger.warning("Too few data points (%d) to compute stats", n_days)
            return stats

        total_ret = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
        stats["total_return"] = total_ret

        years = n_days / _TRADING_DAYS_PER_YEAR
        cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        stats["cagr"] = cagr

        ann_vol = float(daily_returns.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
        stats["annualized_volatility"] = ann_vol

        daily_rf = rf / _TRADING_DAYS_PER_YEAR
        excess_daily = daily_returns - daily_rf
        sharpe = (
            float(excess_daily.mean() / excess_daily.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
            if excess_daily.std() > 1e-12
            else 0.0
        )
        stats["sharpe_ratio"] = sharpe

        downside = excess_daily[excess_daily < 0]
        downside_std = float(np.sqrt((downside**2).mean())) * np.sqrt(_TRADING_DAYS_PER_YEAR)
        sortino = (
            float((excess_daily.mean() * _TRADING_DAYS_PER_YEAR) / downside_std)
            if downside_std > 1e-12
            else 0.0
        )
        stats["sortino_ratio"] = sortino

        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak
        max_dd = float(dd.min())
        stats["max_drawdown"] = max_dd

        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0
        stats["calmar_ratio"] = calmar

        stats["win_rate_daily"] = float((daily_returns > 0).mean())

        weekly_returns = _resample_weekly(daily_returns)
        stats["win_rate_weekly"] = (
            float((weekly_returns > 0).mean()) if len(weekly_returns) > 0 else 0.0
        )

        if turnover is not None and len(turnover) > 0:
            stats["avg_turnover"] = float(turnover.mean())

        if benchmark_equity is not None and len(benchmark_equity) > 1:
            bench_ret = benchmark_equity.pct_change().dropna()
            common = daily_returns.index.intersection(bench_ret.index)
            if len(common) > 1:
                strat = daily_returns.reindex(common).fillna(0.0)
                bench = bench_ret.reindex(common).fillna(0.0)
                excess = strat - bench
                te = float(excess.std() * np.sqrt(_TRADING_DAYS_PER_YEAR))
                ann_excess = float(excess.mean() * _TRADING_DAYS_PER_YEAR)
                stats["excess_return_annualized"] = ann_excess
                stats["information_ratio"] = (
                    ann_excess / te if te > 1e-12 else 0.0
                )

        logger.info(
            "Stats computed: CAGR=%.2f%%  Sharpe=%.2f  MaxDD=%.2f%%",
            cagr * 100,
            sharpe,
            max_dd * 100,
        )
        return stats

    @staticmethod
    def format_stats(stats: dict[str, float]) -> pd.DataFrame:
        """Format raw stats into a presentation-ready DataFrame.

        Returns
        -------
        pd.DataFrame
            Two columns: ``Metric`` and ``Value``.
        """
        _pct = lambda v: f"{v * 100:+.2f}%"  # noqa: E731
        _ratio = lambda v: f"{v:.3f}"  # noqa: E731
        _pct_unsigned = lambda v: f"{v * 100:.2f}%"  # noqa: E731

        formatters: dict[str, Any] = {
            "total_return": ("Total Return", _pct),
            "cagr": ("CAGR", _pct),
            "annualized_volatility": ("Annualized Volatility", _pct_unsigned),
            "sharpe_ratio": ("Sharpe Ratio", _ratio),
            "sortino_ratio": ("Sortino Ratio", _ratio),
            "max_drawdown": ("Max Drawdown", _pct),
            "calmar_ratio": ("Calmar Ratio", _ratio),
            "win_rate_daily": ("Daily Win Rate", _pct_unsigned),
            "win_rate_weekly": ("Weekly Win Rate", _pct_unsigned),
            "avg_turnover": ("Avg Turnover (one-way)", _pct_unsigned),
            "excess_return_annualized": ("Excess Return (ann.)", _pct),
            "information_ratio": ("Information Ratio", _ratio),
        }

        rows: list[dict[str, str]] = []
        for key, (label, fmt_fn) in formatters.items():
            if key in stats:
                rows.append({"Metric": label, "Value": fmt_fn(stats[key])})

        return pd.DataFrame(rows)


def _resample_weekly(daily_returns: pd.Series) -> pd.Series:
    """Compound daily returns into weekly returns (Friday-to-Friday)."""
    idx = pd.DatetimeIndex(daily_returns.index)
    temp = daily_returns.copy()
    temp.index = idx
    weekly = temp.resample("W-FRI").apply(lambda x: (1 + x).prod() - 1)
    return weekly
