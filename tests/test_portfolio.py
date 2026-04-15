"""Unit tests for portfolio construction and rebalancing."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.portfolio.construction import PortfolioConstructor


# ── Fixtures ─────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
REBALANCE_DATE = datetime.date(2024, 6, 28)


@pytest.fixture()
def base_cfg() -> dict:
    return {
        "portfolio": {
            "top_n": 5,
            "max_position_weight": 0.30,
            "equal_weight": True,
            "allow_cash": True,
            "transaction_cost_bps": 10,
            "slippage_bps": 5,
        },
        "rebalance": {
            "frequency": "weekly",
            "day_of_week": 4,
        },
        "dates": {
            "start": "2024-01-01",
            "end": "2024-12-31",
        },
    }


@pytest.fixture()
def ranked_df() -> pd.DataFrame:
    """Ranked DataFrame for a single date with 10 tickers."""
    rng = np.random.default_rng(99)
    scores = rng.uniform(-1, 1, size=len(TICKERS))
    ranks = pd.Series(scores).rank(ascending=False, method="min").astype(int).values
    return pd.DataFrame({
        "date": pd.Timestamp(REBALANCE_DATE),
        "ticker": TICKERS,
        "composite_score": scores,
        "composite_rank": ranks,
    })


# ── Tests ────────────────────────────────────────────────────────────

class TestSelectHoldings:
    def test_returns_correct_count(
        self, base_cfg: dict, ranked_df: pd.DataFrame
    ) -> None:
        constructor = PortfolioConstructor(base_cfg)
        holdings = constructor.select_holdings(ranked_df, REBALANCE_DATE)
        stock_holdings = holdings[holdings["ticker"] != "_CASH"]
        assert len(stock_holdings) == base_cfg["portfolio"]["top_n"]

    def test_weights_sum_to_one(
        self, base_cfg: dict, ranked_df: pd.DataFrame
    ) -> None:
        constructor = PortfolioConstructor(base_cfg)
        holdings = constructor.select_holdings(ranked_df, REBALANCE_DATE)
        assert abs(holdings["weight"].sum() - 1.0) < 1e-9

    def test_max_position_weight_enforced(
        self, base_cfg: dict, ranked_df: pd.DataFrame
    ) -> None:
        base_cfg["portfolio"]["max_position_weight"] = 0.25
        constructor = PortfolioConstructor(base_cfg)
        holdings = constructor.select_holdings(ranked_df, REBALANCE_DATE)
        stock_weights = holdings.loc[
            holdings["ticker"] != "_CASH", "weight"
        ]
        assert (stock_weights <= 0.25 + 1e-9).all(), (
            f"Weight cap violated: {stock_weights.max():.4f}"
        )

    def test_fewer_stocks_than_top_n(self, base_cfg: dict) -> None:
        """When fewer stocks are available, cash fills the gap."""
        small_df = pd.DataFrame({
            "date": pd.Timestamp(REBALANCE_DATE),
            "ticker": ["AAPL", "MSFT"],
            "composite_score": [0.9, 0.5],
            "composite_rank": [1, 2],
        })
        base_cfg["portfolio"]["top_n"] = 5
        constructor = PortfolioConstructor(base_cfg)
        holdings = constructor.select_holdings(small_df, REBALANCE_DATE)
        assert abs(holdings["weight"].sum() - 1.0) < 1e-9
        assert "_CASH" in holdings["ticker"].values


class TestComputeTurnover:
    def test_identical_portfolios_zero_turnover(self) -> None:
        w = {"AAPL": 0.5, "MSFT": 0.5}
        assert PortfolioConstructor.compute_turnover(w, w) == pytest.approx(0.0)

    def test_complete_turnover(self) -> None:
        prev = {"AAPL": 0.5, "MSFT": 0.5}
        new = {"GOOG": 0.5, "AMZN": 0.5}
        assert PortfolioConstructor.compute_turnover(prev, new) == pytest.approx(1.0)

    def test_partial_turnover(self) -> None:
        prev = {"AAPL": 0.5, "MSFT": 0.5}
        new = {"AAPL": 0.5, "GOOG": 0.5}
        to = PortfolioConstructor.compute_turnover(prev, new)
        assert to == pytest.approx(0.5)

    def test_empty_to_full(self) -> None:
        prev: dict[str, float] = {}
        new = {"AAPL": 0.5, "MSFT": 0.5}
        to = PortfolioConstructor.compute_turnover(prev, new)
        assert to == pytest.approx(0.5)
