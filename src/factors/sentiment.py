"""Sentiment factor — **placeholder** for future NLP / news integration.

This module returns a constant (zero) sentiment score for every
date-ticker pair.  Once a real sentiment pipeline is available (e.g.
FinBERT on earnings-call transcripts, or a news-API sentiment feed),
replace the body of :meth:`SentimentFactor.compute` with actual logic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.factors.base import BaseFactor

logger = logging.getLogger(__name__)


class SentimentFactor(BaseFactor):
    """Placeholder sentiment factor.

    Generates a ``sentiment_score`` column filled with small random noise
    (mean 0, std 0.01) so downstream code can be tested end-to-end
    without a live NLP pipeline.

    .. note::
       Replace this stub with real sentiment data when available.

    Returns
    -------
    DataFrame
        Columns: ``date, ticker, sentiment_score``.
    """

    name: str = "sentiment"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing sentiment factors (PLACEHOLDER)")
        df = prices[["date", "ticker"]].copy()

        rng = np.random.default_rng(self._seed)
        df["sentiment_score"] = rng.normal(loc=0.0, scale=0.01, size=len(df))

        logger.info("Sentiment factors complete — %d rows (mock data)", len(df))
        return df
