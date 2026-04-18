# Multi-Factor Strategy — Backtest Report
*Generated 2026-04-17 02:04 UTC*

## Performance Summary

| Metric | Value |
| --- | --- |
| Total Return | +301.28% |
| CAGR | +26.55% |
| Annualized Volatility | 24.67% |
| Sharpe Ratio | 1.078 |
| Sortino Ratio | 1.035 |
| Max Drawdown | -27.37% |
| Calmar Ratio | 0.970 |
| Daily Win Rate | 55.75% |
| Weekly Win Rate | 57.10% |
| Avg Turnover (one-way) | 47.78% |
| Excess Return (ann.) | +10.38% |
| Information Ratio | 0.689 |

## Key Highlights

- **CAGR**: +26.55%
- **Sharpe Ratio**: 1.078
- **Max Drawdown**: -27.37%
- **Sortino Ratio**: 1.035
- **Calmar Ratio**: 0.970

## Configuration

```yaml
universe:
  source: csv
  csv_path: data/sp500_universe.csv
  watchlist: []
  min_history_days: 252
  path: data/sp500_universe.csv
dates:
  start: 2020-01-01
  end: 2025-12-31
rebalance:
  frequency: monthly
  day_of_week: 4
portfolio:
  top_n: 10
  max_position_weight: 0.1
  equal_weight: True
  transaction_cost_bps: 10
  slippage_bps: 5
  allow_cash: True
benchmark:
  ticker: SPY
factors:
  momentum:
    enabled: True
    weights:
      return_1m: 0.15
      return_3m: 0.35
      return_6m: 0.2
    direction: higher_is_better
  trend:
    enabled: True
    weights:
      dist_ma50: 0.15
      dist_ma200: 0.1
    direction: higher_is_better
  mean_reversion:
    enabled: False
    weights:
      rsi_14: 0.05
    direction: lower_is_better
  liquidity:
    enabled: True
    weights:
      avg_dollar_volume: 0.05
      relative_volume: 0.0
    direction: higher_is_better
  volatility:
    enabled: False
    weights:
      realized_vol_60d: 0.05
    direction: lower_is_better
  fundamental:
    enabled: False
    weights:
      pe_ratio: 0.05
      ebitda_margin: 0.05
    direction: mixed
  sentiment:
    enabled: False
    weights:
      sentiment_score: 0.0
    direction: higher_is_better
output:
  charts_dir: outputs/charts
  tables_dir: outputs/tables
  reports_dir: outputs/reports
  chart_format: png
  dpi: 150
  theme: light
logging:
  level: INFO
  file: outputs/pipeline.log
research:
  sector_neutral: False
  weighting_scheme: static
  winsorize_lower: 0.01
  winsorize_upper: 0.99
sensitivity:
  top_n_values: [5, 10, 15, 20, 30, 50]
  cost_values: [0, 5, 10, 20, 30, 50]
  frequencies: ['weekly', 'monthly']
```

## Holdings at Last Rebalance

*Date: 2025-12-30*

| ticker | weight |
| --- | --- |
| MU | 10.00% |
| LRCX | 10.00% |
| GOOGL | 10.00% |
| LLY | 10.00% |
| REGN | 10.00% |
| MRK | 10.00% |
| AMD | 10.00% |
| CAT | 10.00% |
| KLAC | 10.00% |
| USB | 10.00% |

## Top / Bottom Performing Periods

**Best months:**

| Month | Return |
|-------|--------|
| 2020-04 | +15.69% |
| 2020-11 | +14.98% |
| 2025-10 | +13.65% |

**Worst months:**

| Month | Return |
|-------|--------|
| 2022-01 | -14.36% |
| 2022-09 | -7.23% |
| 2023-09 | -7.13% |

## Assumptions & Caveats

- Returns are computed on a **close-to-close** basis using adjusted prices.
- Transaction costs and slippage are modelled as fixed basis-point charges per unit of turnover.
- No market-impact model is applied; results may overstate achievable returns for large portfolios.
- Short selling is **not** modelled; the strategy is long-only.
- Survivorship bias may be present depending on the underlying data source.
- Factor scores are computed from point-in-time data but look-ahead bias in fundamental data cannot be fully excluded.
- Past performance is not indicative of future results.
