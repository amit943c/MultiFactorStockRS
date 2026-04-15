# Research Diagnostics Report

*Generated 2026-04-14 23:21 UTC*

# Validation Report
*Generated 2026-04-14 23:21 UTC*

## 1. Summary

| Result | Count |
|--------|-------|
| PASS   | 5     |
| WARN   | 2     |
| FAIL   | 0     |

**Overall: ISSUES DETECTED**

## 2. Lookahead Bias Checks

| Check | Status | Detail |
|-------|--------|--------|
| factor_timing | PASS | All factor dates exist in prices; warm-up periods contain NaN as expected. No evidence of lookahead bias in factor timing. |
| no_future_data_in_factors | WARN | Cannot run future-data check: need momentum columns and adj_close in factor_df. |
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
| missing_data | WARN | 3 column(s) with >10% missing: dist_ma200 (14.1%), composite_score (14.1%), composite_rank (14.1%) |

## Warnings

- [no_future_data_in_factors] Cannot run future-data check: need momentum columns and adj_close in factor_df.
- [missing_data] 3 column(s) with >10% missing: dist_ma200 (14.1%), composite_score (14.1%), composite_rank (14.1%)


## Factor IC Summary

| factor            |      mean_ic |    ic_std |       ic_ir |   hit_rate |   n_periods |
|:------------------|-------------:|----------:|------------:|-----------:|------------:|
| avg_dollar_volume |  0.000753058 | 0.154102  |  0.00488676 |   0.490579 |        1486 |
| relative_volume   | -0.0403329   | 0.134471  | -0.299938   |   0.376178 |        1486 |
| dist_ma50         |  0.853428    | 0.0889833 |  9.59088    |   1        |        1458 |
| dist_ma200        |  0.497526    | 0.181147  |  2.74652    |   0.980122 |        1308 |

## Score Persistence

- Average score rank autocorrelation: **nan**
- Min: nan, Max: nan

## Holdings Overlap

- Average Jaccard overlap: **1.0000**
- Min: 1.0000, Max: 1.0000

## Drawdown Episodes

| start      | trough     | end        |   depth_pct |   duration_days |   recovery_days |
|:-----------|:-----------|:-----------|------------:|----------------:|----------------:|
| 2020-02-21 | 2020-03-23 | 2020-06-05 |    -27.371  |             105 |              74 |
| 2021-12-30 | 2022-10-14 | 2023-03-21 |    -21.1375 |             446 |             158 |
| 2024-12-18 | 2025-04-08 | 2025-09-18 |    -20.7536 |             274 |             163 |
| 2024-07-11 | 2024-08-07 | 2024-12-11 |    -17.1545 |             153 |             126 |
| 2023-07-20 | 2023-10-27 | 2023-12-11 |    -14.214  |             144 |              45 |
