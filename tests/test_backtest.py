"""Unit tests for the BacktestEngine."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.portfolio.construction import PortfolioConstructor
from src.portfolio.rebalance import RebalanceEngine, RebalanceHistory


# ── Helpers ──────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
N_DAYS = 200


def _make_prices(n_days: int = N_DAYS) -> pd.DataFrame:
    """Deterministic synthetic price panel."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    frames: list[pd.DataFrame] = []
    for ticker in TICKERS:
        base = rng.uniform(80, 400)
        log_ret = rng.normal(0.0004, 0.015, size=n_days)
        close = base * np.exp(np.cumsum(log_ret))
        frames.append(
            pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(1_000_000, 10_000_000, size=n_days).astype(float),
                "adj_close": close,
            })
        )
    return pd.concat(frames, ignore_index=True)


def _make_ranked_df(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Assign random scores so we have composite_score & composite_rank."""
    rng = np.random.default_rng(12)
    unique_dates = prices_df["date"].unique()
    records = []
    for d in unique_dates:
        for t in TICKERS:
            records.append({"date": d, "ticker": t, "composite_score": rng.uniform(-1, 1)})
    df = pd.DataFrame(records)
    df["composite_rank"] = df.groupby("date")["composite_score"].rank(
        ascending=False, method="min",
    )
    return df


def _build_rebalance_history(
    ranked_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cfg: dict,
) -> RebalanceHistory:
    constructor = PortfolioConstructor(cfg)
    engine = RebalanceEngine(constructor)
    return engine.run(ranked_df, prices_df, cfg)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> dict:
    return {
        "portfolio": {
            "top_n": 3,
            "max_position_weight": 0.50,
            "equal_weight": True,
            "allow_cash": True,
            "transaction_cost_bps": 10,
            "slippage_bps": 5,
        },
        "rebalance": {
            "frequency": "monthly",
            "day_of_week": 4,
        },
        "dates": {
            "start": str(pd.bdate_range(end="2024-12-31", periods=N_DAYS)[0].date()),
            "end": "2024-12-31",
        },
    }


@pytest.fixture()
def prices_df() -> pd.DataFrame:
    return _make_prices()


@pytest.fixture()
def ranked_df(prices_df: pd.DataFrame) -> pd.DataFrame:
    return _make_ranked_df(prices_df)


@pytest.fixture()
def rebalance_history(
    ranked_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cfg: dict,
) -> RebalanceHistory:
    return _build_rebalance_history(ranked_df, prices_df, cfg)


# ── Tests ────────────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_equity_curve_starts_near_one(
        self,
        cfg: dict,
        rebalance_history: RebalanceHistory,
        prices_df: pd.DataFrame,
    ) -> None:
        bt = BacktestEngine(cfg)
        result = bt.run(rebalance_history, prices_df)
        first_val = result.equity_curve.iloc[0]
        assert first_val == pytest.approx(1.0, abs=0.02)

    def test_output_shape(
        self,
        cfg: dict,
        rebalance_history: RebalanceHistory,
        prices_df: pd.DataFrame,
    ) -> None:
        bt = BacktestEngine(cfg)
        result = bt.run(rebalance_history, prices_df)
        assert len(result.equity_curve) == len(result.daily_returns)
        assert len(result.equity_curve) > 0

    def test_gross_vs_net_costs(
        self,
        cfg: dict,
        rebalance_history: RebalanceHistory,
        prices_df: pd.DataFrame,
    ) -> None:
        """Net equity should be ≤ gross equity because of transaction costs."""
        bt = BacktestEngine(cfg)
        result = bt.run(rebalance_history, prices_df)
        assert result.equity_curve.iloc[-1] <= result.equity_curve_gross.iloc[-1] + 1e-9

    def test_zero_costs_match_gross(
        self,
        rebalance_history: RebalanceHistory,
        prices_df: pd.DataFrame,
    ) -> None:
        zero_cost_cfg = {
            "portfolio": {
                "transaction_cost_bps": 0,
                "slippage_bps": 0,
            },
        }
        bt = BacktestEngine(zero_cost_cfg)
        result = bt.run(rebalance_history, prices_df)
        pd.testing.assert_series_equal(
            result.equity_curve,
            result.equity_curve_gross,
            check_names=False,
            atol=1e-10,
        )

    def test_result_has_turnover(
        self,
        cfg: dict,
        rebalance_history: RebalanceHistory,
        prices_df: pd.DataFrame,
    ) -> None:
        bt = BacktestEngine(cfg)
        result = bt.run(rebalance_history, prices_df)
        assert len(result.turnover) > 0
        assert (result.turnover >= 0).all()

    def test_result_has_rebalance_dates(
        self,
        cfg: dict,
        rebalance_history: RebalanceHistory,
        prices_df: pd.DataFrame,
    ) -> None:
        bt = BacktestEngine(cfg)
        result = bt.run(rebalance_history, prices_df)
        assert len(result.rebalance_dates) > 0
        assert all(isinstance(d, datetime.date) for d in result.rebalance_dates)

    def test_empty_rebalance_history(self, cfg: dict, prices_df: pd.DataFrame) -> None:
        empty = RebalanceHistory()
        bt = BacktestEngine(cfg)
        result = bt.run(empty, prices_df)
        assert len(result.equity_curve) == 0
