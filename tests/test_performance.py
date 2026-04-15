"""Unit tests for the PerformanceAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.performance import PerformanceAnalyzer


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def flat_equity() -> tuple[pd.Series, pd.Series]:
    """Equity curve with zero return every day (edge-case test)."""
    dates = pd.bdate_range("2023-01-02", periods=252)
    equity = pd.Series(1.0, index=dates.date)
    equity.name = "equity"
    returns = pd.Series(0.0, index=dates.date)
    returns.name = "return"
    return equity, returns


@pytest.fixture()
def known_equity() -> tuple[pd.Series, pd.Series]:
    """Equity curve that doubles over ~252 trading days (≈100 % return)."""
    dates = pd.bdate_range("2023-01-02", periods=252)
    daily_ret = np.full(252, np.log(2) / 252)
    returns = pd.Series(np.exp(daily_ret) - 1, index=dates.date, name="return")
    equity = (1 + returns).cumprod()
    equity.name = "equity"
    return equity, returns


@pytest.fixture()
def drawdown_equity() -> tuple[pd.Series, pd.Series]:
    """Equity curve that rises to 1.5 then drops to 1.0 (33 % drawdown)."""
    n_up, n_down = 100, 50
    dates = pd.bdate_range("2023-01-02", periods=n_up + n_down)

    up_daily = (1.5 ** (1 / n_up)) - 1
    down_daily = ((1.0 / 1.5) ** (1 / n_down)) - 1

    daily_ret = np.concatenate([
        np.full(n_up, up_daily),
        np.full(n_down, down_daily),
    ])
    returns = pd.Series(daily_ret, index=dates.date, name="return")
    equity = (1 + returns).cumprod()
    equity.name = "equity"
    return equity, returns


# ── Tests ────────────────────────────────────────────────────────────

class TestPerformanceStats:
    def test_all_expected_keys(self, known_equity: tuple) -> None:
        equity, returns = known_equity
        stats = PerformanceAnalyzer.compute_stats(equity, returns)
        expected_keys = {
            "total_return",
            "cagr",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate_daily",
            "win_rate_weekly",
        }
        assert expected_keys <= set(stats), (
            f"Missing keys: {expected_keys - set(stats)}"
        )

    def test_cagr_approximately_100pct(self, known_equity: tuple) -> None:
        equity, returns = known_equity
        stats = PerformanceAnalyzer.compute_stats(equity, returns)
        assert stats["cagr"] == pytest.approx(1.0, abs=0.05)

    def test_total_return_approximately_100pct(self, known_equity: tuple) -> None:
        equity, returns = known_equity
        stats = PerformanceAnalyzer.compute_stats(equity, returns)
        assert stats["total_return"] == pytest.approx(1.0, abs=0.02)

    def test_sharpe_positive_for_positive_returns(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=252)
        rng = np.random.default_rng(42)
        daily_ret = 0.002 + 0.005 * rng.standard_normal(252)
        returns = pd.Series(daily_ret, index=dates.date, name="return")
        equity = (1 + returns).cumprod()
        equity.name = "equity"
        stats = PerformanceAnalyzer.compute_stats(equity, returns)
        assert stats["sharpe_ratio"] > 0

    def test_max_drawdown_for_known_series(self, drawdown_equity: tuple) -> None:
        equity, returns = drawdown_equity
        stats = PerformanceAnalyzer.compute_stats(equity, returns)
        assert stats["max_drawdown"] == pytest.approx(-1 / 3, abs=0.02)

    def test_flat_curve_zero_returns(self, flat_equity: tuple) -> None:
        equity, returns = flat_equity
        stats = PerformanceAnalyzer.compute_stats(equity, returns)
        assert stats["total_return"] == pytest.approx(0.0, abs=1e-9)
        assert stats["cagr"] == pytest.approx(0.0, abs=1e-9)
        assert stats["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_benchmark_relative_metrics(self, known_equity: tuple) -> None:
        equity, returns = known_equity
        bench = pd.Series(1.0, index=equity.index, name="bench")
        stats = PerformanceAnalyzer.compute_stats(
            equity, returns, benchmark_equity=bench,
        )
        assert "excess_return_annualized" in stats
        assert "information_ratio" in stats
        assert stats["excess_return_annualized"] > 0

    def test_turnover_metric(self, known_equity: tuple) -> None:
        equity, returns = known_equity
        turnover = pd.Series(
            [0.10, 0.15, 0.20],
            index=pd.bdate_range("2023-02-01", periods=3).date,
        )
        stats = PerformanceAnalyzer.compute_stats(
            equity, returns, turnover=turnover,
        )
        assert "avg_turnover" in stats
        assert stats["avg_turnover"] == pytest.approx(0.15, abs=1e-9)


class TestFormatStats:
    def test_format_returns_dataframe(self, known_equity: tuple) -> None:
        equity, returns = known_equity
        stats = PerformanceAnalyzer.compute_stats(equity, returns)
        table = PerformanceAnalyzer.format_stats(stats)
        assert isinstance(table, pd.DataFrame)
        assert "Metric" in table.columns
        assert "Value" in table.columns
        assert len(table) > 0
