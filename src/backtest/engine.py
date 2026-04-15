"""Backtest engine — walk through calendar days, apply portfolio weights, compute returns.

Execution assumption
--------------------
On each rebalance date *t*, the portfolio weights are determined using data
available up to and including *t*.  Trades are assumed to execute at *t*'s
closing price (adj_close).  This is a **close-to-close** model: the first
return accrues from the close of day *t* to the close of day *t+1*.

Between rebalances the portfolio follows a **buy-and-hold drift** — weights
evolve with each stock's daily return, meaning no implicit daily re-weighting.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.rebalance import RebalanceHistory

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for all backtest outputs.

    Attributes
    ----------
    daily_returns:
        Net daily portfolio returns (after transaction & slippage costs).
    equity_curve:
        Cumulative equity curve starting at 1.0 (net of costs).
    daily_returns_gross:
        Daily returns *before* costs.
    equity_curve_gross:
        Cumulative equity curve before costs.
    holdings_history:
        The :pyattr:`RebalanceHistory.holdings` dict passed through for
        downstream analysis.
    turnover:
        Per-rebalance one-way turnover.
    rebalance_dates:
        Ordered rebalance dates.
    """

    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns_gross: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_curve_gross: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    holdings_history: dict[datetime.date, pd.DataFrame] = field(default_factory=dict)
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rebalance_dates: list[datetime.date] = field(default_factory=list)


class BacktestEngine:
    """Run a daily mark-to-market backtest with buy-and-hold drift.

    Parameters
    ----------
    cfg:
        Application config dict. Reads ``portfolio.transaction_cost_bps`` and
        ``portfolio.slippage_bps``.
    """

    CASH_TICKER = "_CASH"

    def __init__(self, cfg: dict[str, Any]) -> None:
        port_cfg = cfg.get("portfolio", {})
        self._tc_bps: float = float(port_cfg.get("transaction_cost_bps", 10))
        self._slip_bps: float = float(port_cfg.get("slippage_bps", 5))
        logger.info(
            "BacktestEngine: tc_bps=%.1f  slippage_bps=%.1f",
            self._tc_bps,
            self._slip_bps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        rebalance_history: RebalanceHistory,
        prices_df: pd.DataFrame,
    ) -> BacktestResult:
        """Execute the backtest.

        Parameters
        ----------
        rebalance_history:
            Output of :pymeth:`RebalanceEngine.run`.
        prices_df:
            Full price panel with ``date, ticker, adj_close``.

        Returns
        -------
        BacktestResult
        """
        if not rebalance_history.rebalance_dates:
            logger.warning("Empty rebalance history — returning empty result")
            return BacktestResult()

        returns_wide = self._build_returns_panel(prices_df)

        reb_dates = rebalance_history.rebalance_dates
        all_dates = sorted(returns_wide.index)
        start_idx = self._find_index(all_dates, reb_dates[0])
        calendar = all_dates[start_idx:]

        reb_set = set(reb_dates)
        reb_ptr = 0

        weights: dict[str, float] = {}
        daily_ret_gross: list[float] = []
        daily_ret_net: list[float] = []
        date_index: list[datetime.date] = []

        for i, day in enumerate(calendar):
            if day in reb_set:
                while reb_ptr < len(reb_dates) and reb_dates[reb_ptr] <= day:
                    reb_ptr += 1
                target_date = reb_dates[reb_ptr - 1]
                new_weights = self._holdings_to_weights(
                    rebalance_history.holdings[target_date]
                )
                to = rebalance_history.turnover.get(target_date, 0.0)
                cost = to * (self._tc_bps + self._slip_bps) / 10_000.0
                weights = new_weights
                is_rebalance = True
            else:
                cost = 0.0
                is_rebalance = False

            if i == 0:
                daily_ret_gross.append(0.0)
                daily_ret_net.append(-cost if is_rebalance else 0.0)
                date_index.append(day)
                weights = self._drift_weights(weights, returns_wide, day)
                continue

            port_ret = self._portfolio_return(weights, returns_wide, day)
            daily_ret_gross.append(port_ret)
            daily_ret_net.append(port_ret - cost)
            date_index.append(day)

            weights = self._drift_weights(weights, returns_wide, day)

        gross_series = pd.Series(daily_ret_gross, index=pd.Index(date_index, name="date"), name="return_gross")
        net_series = pd.Series(daily_ret_net, index=pd.Index(date_index, name="date"), name="return_net")
        eq_gross = (1.0 + gross_series).cumprod()
        eq_gross.name = "equity_gross"
        eq_net = (1.0 + net_series).cumprod()
        eq_net.name = "equity_net"

        logger.info(
            "Backtest done: %d days, final equity (net)=%.4f, final equity (gross)=%.4f",
            len(calendar),
            eq_net.iloc[-1] if len(eq_net) else 0.0,
            eq_gross.iloc[-1] if len(eq_gross) else 0.0,
        )

        return BacktestResult(
            daily_returns=net_series,
            equity_curve=eq_net,
            daily_returns_gross=gross_series,
            equity_curve_gross=eq_gross,
            holdings_history=rebalance_history.holdings,
            turnover=rebalance_history.turnover,
            rebalance_dates=rebalance_history.rebalance_dates,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_returns_panel(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot adj_close to wide format and compute daily returns.

        Returns a DataFrame indexed by ``date`` (as ``datetime.date``) with
        one column per ticker containing simple daily returns.
        """
        pivot = prices_df.pivot_table(
            index="date", columns="ticker", values="adj_close"
        )
        pivot.index = pd.to_datetime(pivot.index).date
        returns = pivot.pct_change()
        return returns

    @staticmethod
    def _find_index(sorted_dates: list[datetime.date], target: datetime.date) -> int:
        """Binary-style search for the first date >= target."""
        for i, d in enumerate(sorted_dates):
            if d >= target:
                return i
        return len(sorted_dates) - 1

    @classmethod
    def _holdings_to_weights(cls, holdings_df: pd.DataFrame) -> dict[str, float]:
        return dict(zip(holdings_df["ticker"], holdings_df["weight"]))

    @classmethod
    def _portfolio_return(
        cls,
        weights: dict[str, float],
        returns_wide: pd.DataFrame,
        day: datetime.date,
    ) -> float:
        """Weighted portfolio return for a single day."""
        if day not in returns_wide.index:
            return 0.0
        day_ret = returns_wide.loc[day]
        total = 0.0
        for ticker, w in weights.items():
            if ticker == cls.CASH_TICKER:
                continue
            r = day_ret.get(ticker, 0.0)
            if np.isnan(r):
                r = 0.0
            total += w * r
        return total

    @classmethod
    def _drift_weights(
        cls,
        weights: dict[str, float],
        returns_wide: pd.DataFrame,
        day: datetime.date,
    ) -> dict[str, float]:
        """Update weights to reflect buy-and-hold drift after one day's return.

        Each position's notional value grows by ``(1 + r_i)`` and the weights
        are re-normalised by the new portfolio value.
        """
        if day not in returns_wide.index:
            return weights

        day_ret = returns_wide.loc[day]
        new_vals: dict[str, float] = {}
        for ticker, w in weights.items():
            if ticker == cls.CASH_TICKER:
                new_vals[ticker] = w
                continue
            r = day_ret.get(ticker, 0.0)
            if np.isnan(r):
                r = 0.0
            new_vals[ticker] = w * (1.0 + r)

        total = sum(new_vals.values())
        if total <= 0:
            return weights
        return {t: v / total for t, v in new_vals.items()}
