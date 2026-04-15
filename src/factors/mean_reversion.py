"""Mean-reversion factor — Relative Strength Index (RSI)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor

logger = logging.getLogger(__name__)

_RSI_PERIOD: int = 14


def _rsi_wilder(close: pd.Series, period: int = _RSI_PERIOD) -> pd.Series:
    """Compute RSI using Wilder's exponential smoothing.

    Parameters
    ----------
    close:
        Ordered price series for a single ticker.
    period:
        Lookback window (default 14).

    Returns
    -------
    Series
        RSI values in [0, 100]; early values are NaN until enough
        history is available.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing is an EWM with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)

    # Mask the warm-up window
    rsi.iloc[: period - 1] = np.nan
    return rsi


class MeanReversionFactor(BaseFactor):
    """14-day RSI computed with Wilder's smoothing.

    Returns
    -------
    DataFrame
        Columns: ``date, ticker, rsi_14``.
    """

    name: str = "mean_reversion"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing mean-reversion factors (RSI-%d)", _RSI_PERIOD)
        df = prices[["date", "ticker", "adj_close"]].copy()
        df = df.sort_values(["ticker", "date"])

        df["rsi_14"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: _rsi_wilder(s, _RSI_PERIOD)
        )

        result = df[["date", "ticker", "rsi_14"]]
        logger.info("Mean-reversion factors complete — %d rows", len(result))
        return result
