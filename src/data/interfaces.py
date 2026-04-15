"""Abstract data-source interface for the multi-factor ranking system.

Concrete implementations (Yahoo Finance, database-backed, CSV-based, etc.)
subclass :class:`DataSource` and implement the two fetch methods.
"""

from __future__ import annotations

import abc
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    """Base class every data provider must implement."""

    @abc.abstractmethod
    def fetch_prices(
        self,
        tickers: list[str],
        start: date | str,
        end: date | str,
    ) -> pd.DataFrame:
        """Download OHLCV price history for *tickers* between *start* and *end*.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns:
            ``date, ticker, open, high, low, close, volume, adj_close``.
        """

    @abc.abstractmethod
    def fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """Retrieve fundamental data for *tickers*.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ``ticker, pe_ratio, ebitda_margin, market_cap, sector``.
        """
