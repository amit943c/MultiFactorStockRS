"""Liquidity factor — dollar-volume and relative volume."""

from __future__ import annotations

import logging

import pandas as pd

from src.factors.base import BaseFactor

logger = logging.getLogger(__name__)

_VOLUME_WINDOW: int = 20


class LiquidityFactor(BaseFactor):
    """Compute 20-day average dollar volume and relative volume.

    * ``avg_dollar_volume``: rolling 20-day mean of ``close * volume``.
    * ``relative_volume``: today's volume divided by its 20-day average.

    Returns
    -------
    DataFrame
        Columns: ``date, ticker, avg_dollar_volume, relative_volume``.
    """

    name: str = "liquidity"

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing liquidity factors (window=%d)", _VOLUME_WINDOW)
        df = prices[["date", "ticker", "close", "volume"]].copy()
        df = df.sort_values(["ticker", "date"])

        df["dollar_volume"] = df["close"] * df["volume"]

        df["avg_dollar_volume"] = df.groupby("ticker")["dollar_volume"].transform(
            lambda s: s.rolling(_VOLUME_WINDOW, min_periods=_VOLUME_WINDOW).mean()
        )

        avg_vol = df.groupby("ticker")["volume"].transform(
            lambda s: s.rolling(_VOLUME_WINDOW, min_periods=_VOLUME_WINDOW).mean()
        )
        df["relative_volume"] = df["volume"] / (avg_vol + 1e-12)

        result = df[["date", "ticker", "avg_dollar_volume", "relative_volume"]]
        logger.info("Liquidity factors complete — %d rows", len(result))
        return result
