"""Volatility factor — realised volatility from log returns."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor

logger = logging.getLogger(__name__)

_VOL_WINDOW: int = 60
_ANNUALISATION: float = np.sqrt(252)


class VolatilityFactor(BaseFactor):
    """60-day realised volatility, annualised.

    Computed as the rolling standard deviation of daily log returns
    multiplied by sqrt(252).

    Returns
    -------
    DataFrame
        Columns: ``date, ticker, realized_vol_60d``.
    """

    name: str = "volatility"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing volatility factors (window=%d)", _VOL_WINDOW)
        df = prices[["date", "ticker", "adj_close"]].copy()
        df = df.sort_values(["ticker", "date"])

        df["log_return"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

        df["realized_vol_60d"] = df.groupby("ticker")["log_return"].transform(
            lambda s: s.rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW).std()
            * _ANNUALISATION
        )

        result = df[["date", "ticker", "realized_vol_60d"]]
        logger.info("Volatility factors complete — %d rows", len(result))
        return result
