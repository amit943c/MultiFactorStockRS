"""Shared numerical helpers used across factor and analytics modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip values to the given quantile bounds."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score *col* within each date cross-section.

    Expects *df* to have a ``date`` column (or DatetimeIndex).
    """
    if "date" in df.columns:
        grouped = df.groupby("date")[col]
    else:
        grouped = df.groupby(df.index.get_level_values(0))[col]
    return grouped.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))


def rank_pct(series: pd.Series) -> pd.Series:
    """Percentile rank within a cross-section (0 to 1)."""
    return series.rank(pct=True)


def annualize_returns(total_return: float, days: int) -> float:
    """Convert a total return over *days* trading days to CAGR."""
    if days <= 0:
        return 0.0
    return (1 + total_return) ** (252 / days) - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve (indexed by date)."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the full drawdown series."""
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak


def rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.0) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    excess = returns - rf / 252
    roll_mean = excess.rolling(window).mean() * 252
    roll_std = excess.rolling(window).std() * np.sqrt(252)
    return roll_mean / (roll_std + 1e-9)
