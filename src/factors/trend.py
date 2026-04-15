"""Trend factor — distance from key moving averages."""

from __future__ import annotations

import logging

import pandas as pd

from src.factors.base import BaseFactor

logger = logging.getLogger(__name__)

_MA_WINDOWS: dict[str, int] = {
    "dist_ma50": 50,
    "dist_ma200": 200,
}


class TrendFactor(BaseFactor):
    """Measure how far ``adj_close`` is from its 50-day and 200-day
    simple moving averages, expressed as a fraction of the MA.

    Returns
    -------
    DataFrame
        Columns: ``date, ticker, dist_ma50, dist_ma200``.
    """

    name: str = "trend"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing trend factors")
        df = prices[["date", "ticker", "adj_close"]].copy()
        df = df.sort_values(["ticker", "date"])

        for col_name, window in _MA_WINDOWS.items():
            ma = df.groupby("ticker")["adj_close"].transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )
            df[col_name] = (df["adj_close"] - ma) / ma
            logger.debug("Computed %s (window=%d)", col_name, window)

        result = df[["date", "ticker"] + list(_MA_WINDOWS.keys())]
        logger.info("Trend factors complete — %d rows", len(result))
        return result
