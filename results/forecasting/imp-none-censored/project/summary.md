# Forecast evaluation: imp-none-censored, project protocol

- **Truth:** raw recorded sales (`demand_total` of the censored demand_daily) for every model, whatever demand
  history it was trained on. Only target days with no stockout between 06:00 and 22:00 are scored.
- **WAPE** = sum|F - A| / sum|A| and **WPE** = sum(F - A) / sum|A| per (origin date, target date) cell,
  averaged over the cells: the dataset paper's mean of the 7 daily values for the test week. WPE sign as in
  the paper: positive = over-forecast, negative = under-forecast. Pooled WAPE is one ratio over all scored
  test rows, for information.
- **Test:** mean ± SD over seeds (number of seeds). **Fold CV:** mean ± SD over all (fold, seed) validation
  runs (number of runs); SD_cv is that SD.
- **Published:** Wang et al. 2025 (arXiv:2505.16319), Table 3. **Pass:** test mean <= published + SD_cv.
  **Stretch:** test mean <= published - SD_cv. **Improvement:** (published - test mean) / SD_cv.
  **A.7.2 literal:** |fold CV mean - published| <= SD_cv.
- WAPE, WPE and PICP in percent; MASE and MPIW (normalized sales units) as plain numbers.

## fc-dlinear-baseline (imp-none-censored, project, point estimated_demand)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.76 ± 0.01 (3) | -7.05 ± 0.08 (3) | 31.74 | 38.24 ± 0.78 (18) | 31.56 / -4.89 | yes | no | -0.25 | no |
| low_sales | 39.60 ± 0.02 (3) | -3.66 ± 0.07 (3) | 39.38 | 45.70 ± 0.67 (18) | 39.78 / -1.03 | yes | no | 0.27 | no |
| high_sales | 26.14 ± 0.00 (3) | -9.46 ± 0.08 (3) | 26.19 | 32.38 ± 0.80 (18) | 25.68 / -7.63 | yes | no | -0.58 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.933 | n/a | n/a | n/a | n/a |
| low_sales | 0.918 | n/a | n/a | n/a | n/a |
| high_sales | 0.972 | n/a | n/a | n/a | n/a |

## fc-dlinear-baseline (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.76 ± 0.01 (3) | -7.05 ± 0.08 (3) | 31.74 | 38.24 ± 0.78 (18) | 31.56 / -4.89 | yes | no | -0.25 | no |
| low_sales | 39.60 ± 0.02 (3) | -3.66 ± 0.07 (3) | 39.38 | 45.70 ± 0.67 (18) | 39.78 / -1.03 | yes | no | 0.27 | no |
| high_sales | 26.14 ± 0.00 (3) | -9.46 ± 0.08 (3) | 26.19 | 32.38 ± 0.80 (18) | 25.68 / -7.63 | yes | no | -0.58 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.933 | n/a | n/a | n/a | n/a |
| low_sales | 0.918 | n/a | n/a | n/a | n/a |
| high_sales | 0.972 | n/a | n/a | n/a | n/a |

## fc-dlinear-plus (imp-none-censored, project, point estimated_demand)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 30.67 ± 0.02 (3) | -11.84 ± 0.08 (3) | 30.70 | 38.51 ± 2.48 (18) | 31.56 / -4.89 | yes | no | 0.36 | no |
| low_sales | 37.97 ± 0.01 (3) | -11.81 ± 0.11 (3) | 37.87 | 46.23 ± 2.89 (18) | 39.78 / -1.03 | yes | no | 0.63 | no |
| high_sales | 25.43 ± 0.02 (3) | -11.89 ± 0.07 (3) | 25.49 | 32.43 ± 2.15 (18) | 25.68 / -7.63 | yes | no | 0.11 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.911 | n/a | n/a | n/a | n/a |
| low_sales | 0.893 | n/a | n/a | n/a | n/a |
| high_sales | 0.961 | n/a | n/a | n/a | n/a |

## fc-tft-quantile (imp-none-censored, project, point estimated_demand)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 34.21 ± 0.77 (3) | -8.10 ± 1.94 (3) | 34.08 | 41.56 ± 4.94 (18) | 31.75 / -7.37 | yes | no | -0.50 | no |
| low_sales | 38.95 ± 0.14 (3) | -0.98 ± 5.94 (3) | 38.78 | 50.87 ± 9.53 (18) | 37.04 / -1.72 | yes | no | -0.20 | no |
| high_sales | 30.81 ± 1.26 (3) | -13.20 ± 1.33 (3) | 30.67 | 34.29 ± 2.84 (18) | 27.95 / -11.46 | no | no | -1.01 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.947 | 89.84 | 1.592 | n/a | n/a |
| low_sales | 0.898 | 91.34 | 1.131 | n/a | n/a |
| high_sales | 1.081 | 85.54 | 2.915 | n/a | n/a |

## fc-tft-quantile (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 33.44 ± 0.27 (3) | -11.90 ± 2.89 (3) | 33.30 | 39.48 ± 2.85 (18) | 31.75 / -7.37 | yes | no | -0.59 | no |
| low_sales | 39.04 ± 1.05 (3) | -9.31 ± 8.06 (3) | 38.89 | 45.12 ± 3.27 (18) | 37.04 / -1.72 | yes | no | -0.61 | no |
| high_sales | 29.39 ± 0.62 (3) | -13.75 ± 1.27 (3) | 29.24 | 35.06 ± 4.11 (18) | 27.95 / -11.46 | yes | no | -0.35 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.954 | 89.84 | 1.592 | n/a | n/a |
| low_sales | 0.917 | 91.34 | 1.131 | n/a | n/a |
| high_sales | 1.053 | 85.54 | 2.915 | n/a | n/a |

## Paired bootstrap against the baseline

Test week, all series; forecasts averaged over seeds first. Each draw resamples series with replacement, the
same draw for both models. Delta = model WAPE - baseline WAPE (negative favours the model), with its 95%
bootstrap interval.

| Model | Imputation run | Point | Baseline | Baseline imputation run | WAPE | Baseline WAPE | Delta | 95% CI | Series | Draws |
|---|---|---|---|---|---|---|---|---|---|---|
| fc-tft-quantile | imp-none-censored | q500 | fc-dlinear-baseline | imp-none-censored | 32.49 | 31.76 | 0.73 | [0.55, 0.92] | 50000 | 1000 |
| fc-dlinear-plus | imp-none-censored | estimated_demand | fc-dlinear-baseline | imp-none-censored | 30.67 | 31.76 | -1.08 | [-1.17, -0.99] | 50000 | 1000 |
