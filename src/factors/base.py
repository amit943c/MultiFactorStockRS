"""Base class for all factor modules."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class BaseFactor(ABC):
    """Abstract factor that computes one or more sub-factor columns.

    Every concrete factor must set ``name`` and implement :meth:`compute`.
    The returned DataFrame must always contain at least ``date`` and
    ``ticker`` columns alongside one or more factor-value columns.
    """

    name: str = "base"

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a long-format DataFrame with columns:
        ``date``, ``ticker``, and one or more factor-value columns.

        Parameters
        ----------
        prices:
            Long-format OHLCV DataFrame sorted by date with columns
            ``date, ticker, open, high, low, close, volume, adj_close``.
        """
        ...

    def validate(self, df: pd.DataFrame) -> bool:
        """Check that *df* contains the minimum required columns."""
        required = {"date", "ticker"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("%s.validate: missing columns %s", self.name, missing)
            return False
        return True
