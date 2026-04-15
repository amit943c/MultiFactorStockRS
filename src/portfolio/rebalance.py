"""Rebalance engine — iterate through time, build holdings snapshots, track turnover."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.portfolio.construction import PortfolioConstructor

logger = logging.getLogger(__name__)


@dataclass
class RebalanceHistory:
    """Container for the full rebalance trajectory.

    Attributes
    ----------
    holdings:
        Mapping of rebalance date to the holdings DataFrame returned by
        :pymethod:`PortfolioConstructor.select_holdings`.
    turnover:
        One-way turnover at each rebalance date.
    rebalance_dates:
        Ordered list of all rebalance dates.
    """

    holdings: dict[datetime.date, pd.DataFrame] = field(default_factory=dict)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rebalance_dates: list[datetime.date] = field(default_factory=list)


class RebalanceEngine:
    """Drive the portfolio constructor across every rebalance date.

    Parameters
    ----------
    constructor:
        A fully configured :class:`PortfolioConstructor` instance.
    """

    def __init__(self, constructor: PortfolioConstructor) -> None:
        self._constructor = constructor

    def run(
        self,
        ranked_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        cfg: dict[str, Any],
    ) -> RebalanceHistory:
        """Execute the rebalance schedule and return the full history.

        Parameters
        ----------
        ranked_df:
            DataFrame with ``date, ticker, composite_score, composite_rank``.
        prices_df:
            Prices DataFrame with ``date, ticker, adj_close`` (and others).
        cfg:
            Application config dict (``dates.start``, ``dates.end`` are read).

        Returns
        -------
        RebalanceHistory
        """
        dates_cfg = cfg.get("dates", {})
        start = str(dates_cfg.get("start", ranked_df["date"].min()))
        end = str(dates_cfg.get("end", ranked_df["date"].max()))

        rebalance_dates = self._constructor.generate_rebalance_dates(
            start, end, prices_df
        )

        if not rebalance_dates:
            logger.warning("No rebalance dates generated — returning empty history")
            return RebalanceHistory()

        holdings_map: dict[datetime.date, pd.DataFrame] = {}
        turnover_map: dict[datetime.date, float] = {}
        prev_weights: dict[str, float] = {}

        for reb_date in rebalance_dates:
            hdf = self._constructor.select_holdings(ranked_df, reb_date)
            holdings_map[reb_date] = hdf

            new_weights = dict(zip(hdf["ticker"], hdf["weight"]))
            to = self._constructor.compute_turnover(prev_weights, new_weights)
            turnover_map[reb_date] = to
            prev_weights = new_weights

            logger.debug(
                "Rebalance %s: %d holdings, turnover=%.4f",
                reb_date,
                len(hdf),
                to,
            )

        turnover_series = pd.Series(turnover_map, name="turnover")
        turnover_series.index.name = "date"

        logger.info(
            "Rebalance complete: %d dates, avg turnover=%.4f",
            len(rebalance_dates),
            turnover_series.mean(),
        )

        return RebalanceHistory(
            holdings=holdings_map,
            turnover=turnover_series,
            rebalance_dates=rebalance_dates,
        )
