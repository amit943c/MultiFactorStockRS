# Research Diagnostics Report

*Generated 2026-04-17 02:04 UTC*

# Validation Report
*Generated 2026-04-17 02:04 UTC*

## 1. Summary

| Result | Count |
|--------|-------|
| PASS   | 7     |
| WARN   | 0     |
| FAIL   | 0     |

**Overall: PASS**

## 2. Lookahead Bias Checks

| Check | Status | Detail |
|-------|--------|--------|
| factor_timing | PASS | All factor dates exist in prices; warm-up periods contain NaN as expected. No evidence of lookahead bias in factor timing. |
| no_future_data_in_factors | PASS | Momentum factors correlate with past price changes, not future ones. No evidence of lookahead in factor construction. |
| rebalance_timing | PASS | All rebalance-date holdings are consistent with factor scores available on or before the rebalance date. |

## 3. Data Quality Checks

| Check | Status | Detail |
|-------|--------|--------|
| price_availability | PASS | All 720 positions have price data on their rebalance date. |
| data_quality | PASS | No quality issues detected across 149193 price rows. |

## 4. Survivorship Bias Assessment

| Check | Status | Detail |
|-------|--------|--------|
| survivorship_bias | PASS | 100.0% of tickers have full history (2020-01-02 to 2025-12-30); 0 entered late, 0 exited early |

## 5. Execution Assumptions

*No execution assumptions recorded.*

## 6. Missing Data Summary

| Check | Status | Detail |
|-------|--------|--------|
| missing_data | PASS | All missing data is in warmup-dependent columns (expected): dist_ma200 (13.2%), composite_score (13.2%), composite_rank (13.2%). Moving-average and multi-month return factors require a lookback window before producing values. |


## Factor IC Summary

| factor            |      mean_ic |    ic_std |       ic_ir |   hit_rate |   n_periods |
|:------------------|-------------:|----------:|------------:|-----------:|------------:|
| avg_dollar_volume |  0.000562984 | 0.155596  |  0.00361825 |   0.494616 |        1486 |
| relative_volume   | -0.0400941   | 0.134204  | -0.298755   |   0.381561 |        1486 |
| dist_ma50         |  0.853229    | 0.0891606 |  9.56958    |   1        |        1458 |
| dist_ma200        |  0.497301    | 0.181144  |  2.74533    |   0.980887 |        1308 |

## Score Persistence

- Average score rank autocorrelation: **0.9753**
- Min: 0.7963, Max: 0.9981

## Holdings Overlap

- Average Jaccard overlap: **1.0000**
- Min: 1.0000, Max: 1.0000

## Drawdown Episodes

| start      | trough     | end        |   depth_pct |   duration_days |   recovery_days |
|:-----------|:-----------|:-----------|------------:|----------------:|----------------:|
| 2020-02-21 | 2020-03-23 | 2020-06-05 |    -27.371  |             105 |              74 |
| 2021-12-30 | 2022-09-30 | 2023-03-21 |    -20.5994 |             446 |             172 |
| 2024-07-11 | 2025-04-08 | 2025-09-18 |    -20.2561 |             434 |             163 |
| 2023-07-20 | 2023-10-27 | 2023-12-11 |    -14.214  |             144 |              45 |
| 2025-10-30 | 2025-11-20 |            |    -12.0032 |              61 |             nan |
