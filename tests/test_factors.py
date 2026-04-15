"""Unit tests for factor computations."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.factors.composite import CompositeScorer
from src.factors.liquidity import LiquidityFactor
from src.factors.mean_reversion import MeanReversionFactor
from src.factors.momentum import MomentumFactor
from src.factors.registry import FactorRegistry
from src.factors.trend import TrendFactor
from src.factors.volatility import VolatilityFactor


# ── Fixtures ─────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
N_DAYS = 300


@pytest.fixture()
def synthetic_prices() -> pd.DataFrame:
    """Generate a reproducible synthetic OHLCV panel (5 tickers, 300 days)."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2024-12-31", periods=N_DAYS)

    frames: list[pd.DataFrame] = []
    for ticker in TICKERS:
        base_price = rng.uniform(50, 500)
        log_returns = rng.normal(0.0003, 0.02, size=N_DAYS)
        close = base_price * np.exp(np.cumsum(log_returns))
        volume = rng.integers(1_000_000, 20_000_000, size=N_DAYS).astype(float)

        df = pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "open": close * rng.uniform(0.99, 1.01, size=N_DAYS),
            "high": close * rng.uniform(1.00, 1.03, size=N_DAYS),
            "low": close * rng.uniform(0.97, 1.00, size=N_DAYS),
            "close": close,
            "volume": volume,
            "adj_close": close,
        })
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ── Momentum factor ─────────────────────────────────────────────────

class TestMomentumFactor:
    def test_output_columns(self, synthetic_prices: pd.DataFrame) -> None:
        result = MomentumFactor().compute(synthetic_prices)
        for col in ("date", "ticker", "return_1m", "return_3m", "return_6m"):
            assert col in result.columns

    def test_no_nan_after_warmup(self, synthetic_prices: pd.DataFrame) -> None:
        result = MomentumFactor().compute(synthetic_prices)
        warmup = 126  # longest lookback
        for ticker in TICKERS:
            sub = result[result["ticker"] == ticker].sort_values("date")
            after_warmup = sub.iloc[warmup:]
            assert after_warmup["return_6m"].notna().all(), (
                f"{ticker}: NaN found after warmup in return_6m"
            )

    def test_return_values_finite(self, synthetic_prices: pd.DataFrame) -> None:
        result = MomentumFactor().compute(synthetic_prices)
        numeric = result[["return_1m", "return_3m", "return_6m"]].dropna()
        assert np.isfinite(numeric.values).all()


# ── Trend factor ─────────────────────────────────────────────────────

class TestTrendFactor:
    def test_output_columns(self, synthetic_prices: pd.DataFrame) -> None:
        result = TrendFactor().compute(synthetic_prices)
        assert "dist_ma50" in result.columns
        assert "dist_ma200" in result.columns

    def test_distance_from_ma_sign(self, synthetic_prices: pd.DataFrame) -> None:
        """dist_ma should be (price - MA) / MA, so values are centred around 0."""
        result = TrendFactor().compute(synthetic_prices)
        valid = result["dist_ma50"].dropna()
        assert len(valid) > 0
        assert valid.min() > -1.0, "dist_ma50 implausibly low"
        assert valid.max() < 5.0, "dist_ma50 implausibly high"

    def test_no_nan_after_warmup(self, synthetic_prices: pd.DataFrame) -> None:
        result = TrendFactor().compute(synthetic_prices)
        warmup = 200
        for ticker in TICKERS:
            sub = result[result["ticker"] == ticker].sort_values("date")
            after_warmup = sub.iloc[warmup:]
            assert after_warmup["dist_ma200"].notna().all()


# ── Mean-reversion factor ────────────────────────────────────────────

class TestMeanReversionFactor:
    def test_rsi_range(self, synthetic_prices: pd.DataFrame) -> None:
        result = MeanReversionFactor().compute(synthetic_prices)
        valid_rsi = result["rsi_14"].dropna()
        assert (valid_rsi >= 0).all(), "RSI below 0"
        assert (valid_rsi <= 100).all(), "RSI above 100"

    def test_output_columns(self, synthetic_prices: pd.DataFrame) -> None:
        result = MeanReversionFactor().compute(synthetic_prices)
        assert {"date", "ticker", "rsi_14"} <= set(result.columns)


# ── Liquidity factor ────────────────────────────────────────────────

class TestLiquidityFactor:
    def test_non_negative(self, synthetic_prices: pd.DataFrame) -> None:
        result = LiquidityFactor().compute(synthetic_prices)
        for col in ("avg_dollar_volume", "relative_volume"):
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} contains negative values"

    def test_output_columns(self, synthetic_prices: pd.DataFrame) -> None:
        result = LiquidityFactor().compute(synthetic_prices)
        assert {"date", "ticker", "avg_dollar_volume", "relative_volume"} <= set(
            result.columns
        )


# ── Volatility factor ───────────────────────────────────────────────

class TestVolatilityFactor:
    def test_non_negative(self, synthetic_prices: pd.DataFrame) -> None:
        result = VolatilityFactor().compute(synthetic_prices)
        valid = result["realized_vol_60d"].dropna()
        assert (valid >= 0).all(), "Realised vol contains negative values"

    def test_output_columns(self, synthetic_prices: pd.DataFrame) -> None:
        result = VolatilityFactor().compute(synthetic_prices)
        assert {"date", "ticker", "realized_vol_60d"} <= set(result.columns)


# ── Composite scorer ────────────────────────────────────────────────

class TestCompositeScorer:
    @pytest.fixture()
    def factor_df(self, synthetic_prices: pd.DataFrame) -> pd.DataFrame:
        """Merge multiple factor outputs for composite scoring."""
        mom = MomentumFactor().compute(synthetic_prices)
        trend = TrendFactor().compute(synthetic_prices)
        mr = MeanReversionFactor().compute(synthetic_prices)
        merged = mom.merge(trend, on=["date", "ticker"], how="inner")
        merged = merged.merge(mr, on=["date", "ticker"], how="inner")
        return merged

    def test_composite_columns(self, factor_df: pd.DataFrame) -> None:
        scorer = CompositeScorer(
            factor_weights={"return_1m": 0.3, "dist_ma50": 0.3, "rsi_14": 0.4},
            factor_directions={
                "return_1m": "higher_is_better",
                "dist_ma50": "higher_is_better",
                "rsi_14": "lower_is_better",
            },
        )
        result = scorer.compute_composite(factor_df)
        assert "composite_score" in result.columns
        assert "composite_rank" in result.columns

    def test_rank_order(self, factor_df: pd.DataFrame) -> None:
        scorer = CompositeScorer(
            factor_weights={"return_1m": 1.0},
            factor_directions={"return_1m": "higher_is_better"},
        )
        result = scorer.compute_composite(factor_df)
        for _, grp in result.groupby("date"):
            valid = grp.dropna(subset=["composite_rank", "composite_score"])
            if len(valid) < 2:
                continue
            top = valid.loc[valid["composite_rank"].idxmin()]
            assert top["composite_score"] == valid["composite_score"].max()

    def test_composite_score_finite(self, factor_df: pd.DataFrame) -> None:
        scorer = CompositeScorer(
            factor_weights={"return_1m": 0.5, "dist_ma50": 0.5},
            factor_directions={
                "return_1m": "higher_is_better",
                "dist_ma50": "higher_is_better",
            },
        )
        result = scorer.compute_composite(factor_df)
        scores = result["composite_score"].dropna()
        assert np.isfinite(scores.values).all()
