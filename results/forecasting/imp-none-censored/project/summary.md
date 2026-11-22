# Forecast evaluation: imp-none-censored, project protocol

- **Truth:** raw recorded sales (`demand_total` of the censored demand_daily) for every model, whatever demand
  history it was trained on. Only target days with no stockout between 06:00 and 22:00 are scored.
- **WAPE** = sum|F - A| / sum|A| and **WPE** = sum(F - A) / sum|A| over all scored rows (the literature
  definition, Dev Contract A.4); each row is one store-product-day, so no errors are netted before the
  absolute value. WPE sign as in the dataset paper: positive = over-forecast, negative = under-forecast.
  **Paper-style WAPE** is the dataset paper's own scoring (the ratio per target day, averaged over the days),
  shown only to compare with its published numbers.
- **Test:** mean ± SD over seeds (number of seeds). **Fold CV:** mean ± SD over all (fold, seed) validation
  runs (number of runs); SD_cv is that SD.
- **Published:** Wang et al. 2025 (arXiv:2505.16319), Table 3. **Pass (A.7.2):** test WAPE <= published +
  SD_cv, with SD_cv from the >= 3-fold cross-validation (Dev Contract, Appendix A.7, item 2).
  **>= 1 SD better:** test WAPE <= published - SD_cv, an optional team goal (not a contract term).
  **Improvement:** (published - test WAPE) / SD_cv. **A.7.2 literal:** |fold CV mean - published| <= SD_cv.
- **PICP / MPIW:** the team functions in `evaluation/uncertainty_metrics.py`: rows with both bounds,
  degenerate intervals (lower = upper, e.g. [0, 0]) left out.
- Metric rows follow `contracts/CONTRACTS.md` §2.10, plus imputation_run_id, protocol, point and n_skipped.
- WAPE, WPE and PICP in percent; MASE and MPIW (normalized sales units) as plain numbers.

## dlinear_forecasting_quantile_ma29.yaml (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | n/a | n/a | n/a | 39.00 ± 0.58 (6) | n/a / n/a | n/a | n/a | n/a | n/a |
| low_sales | n/a | n/a | n/a | 45.12 ± 0.54 (6) | n/a / n/a | n/a | n/a | n/a | n/a |
| high_sales | n/a | n/a | n/a | 33.29 ± 1.15 (6) | n/a / n/a | n/a | n/a | n/a | n/a |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | n/a | n/a | n/a | n/a | n/a |
| low_sales | n/a | n/a | n/a | n/a | n/a |
| high_sales | n/a | n/a | n/a | n/a | n/a |

## fc-dlinear-baseline (imp-none-censored, project, point estimated_demand)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.74 ± 0.01 (3) | -7.38 ± 0.08 (3) | 31.76 | 38.06 ± 0.80 (18) | 31.56 / -4.89 | yes | no | -0.23 | no |
| low_sales | 39.38 ± 0.02 (3) | -4.34 ± 0.08 (3) | 39.60 | 45.35 ± 0.65 (18) | 39.78 / -1.03 | yes | no | 0.62 | no |
| high_sales | 26.19 ± 0.00 (3) | -9.60 ± 0.08 (3) | 26.14 | 32.30 ± 0.84 (18) | 25.68 / -7.63 | yes | no | -0.61 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.933 | n/a | n/a | n/a | n/a |
| low_sales | 0.918 | n/a | n/a | n/a | n/a |
| high_sales | 0.972 | n/a | n/a | n/a | n/a |

## fc-dlinear-baseline (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.74 ± 0.01 (3) | -7.38 ± 0.08 (3) | 31.76 | 38.06 ± 0.80 (18) | 31.56 / -4.89 | yes | no | -0.23 | no |
| low_sales | 39.38 ± 0.02 (3) | -4.34 ± 0.08 (3) | 39.60 | 45.35 ± 0.65 (18) | 39.78 / -1.03 | yes | no | 0.62 | no |
| high_sales | 26.19 ± 0.00 (3) | -9.60 ± 0.08 (3) | 26.14 | 32.30 ± 0.84 (18) | 25.68 / -7.63 | yes | no | -0.61 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.933 | n/a | n/a | n/a | n/a |
| low_sales | 0.918 | n/a | n/a | n/a | n/a |
| high_sales | 0.972 | n/a | n/a | n/a | n/a |

## fc-dlinear-plus (imp-none-censored, project, point estimated_demand)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 30.70 ± 0.02 (3) | -11.89 ± 0.09 (3) | 30.67 | 38.23 ± 2.26 (18) | 31.56 / -4.89 | yes | no | 0.38 | no |
| low_sales | 37.87 ± 0.01 (3) | -11.87 ± 0.11 (3) | 37.97 | 45.80 ± 2.62 (18) | 39.78 / -1.03 | yes | no | 0.73 | no |
| high_sales | 25.49 ± 0.02 (3) | -11.91 ± 0.08 (3) | 25.43 | 32.26 ± 1.97 (18) | 25.68 / -7.63 | yes | no | 0.10 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.911 | n/a | n/a | n/a | n/a |
| low_sales | 0.893 | n/a | n/a | n/a | n/a |
| high_sales | 0.961 | n/a | n/a | n/a | n/a |

## fc-dlinear-plus (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 30.70 ± 0.02 (3) | -11.89 ± 0.09 (3) | 30.67 | 38.23 ± 2.26 (18) | 31.56 / -4.89 | yes | no | 0.38 | no |
| low_sales | 37.87 ± 0.01 (3) | -11.87 ± 0.11 (3) | 37.97 | 45.80 ± 2.62 (18) | 39.78 / -1.03 | yes | no | 0.73 | no |
| high_sales | 25.49 ± 0.02 (3) | -11.91 ± 0.08 (3) | 25.43 | 32.26 ± 1.97 (18) | 25.68 / -7.63 | yes | no | 0.10 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.911 | n/a | n/a | n/a | n/a |
| low_sales | 0.893 | n/a | n/a | n/a | n/a |
| high_sales | 0.961 | n/a | n/a | n/a | n/a |

## fc-dlinear-quantile (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.91 (1) | -2.54 (1) | 31.92 | 39.05 ± 0.62 (6) | 31.56 / -4.89 | yes | no | -0.56 | no |
| low_sales | 38.44 (1) | -0.06 (1) | 38.61 | 45.07 ± 0.56 (6) | 39.78 / -1.03 | yes | yes | 2.42 | no |
| high_sales | 25.56 (1) | -4.95 (1) | 25.50 | 33.42 ± 1.15 (6) | 25.68 / -7.63 | yes | no | 0.11 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.936 | 95.85 | 2.097 | n/a | n/a |
| low_sales | 0.946 | 96.46 | 1.606 | n/a | n/a |
| high_sales | 0.907 | 93.77 | 3.761 | n/a | n/a |

## fc-tft-quantile (imp-none-censored, project, point estimated_demand)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 34.08 ± 0.74 (3) | -7.87 ± 1.97 (3) | 34.21 | 41.26 ± 4.76 (18) | 31.75 / -7.37 | yes | no | -0.49 | no |
| low_sales | 38.78 ± 0.13 (3) | -0.91 ± 5.93 (3) | 38.95 | 50.24 ± 9.20 (18) | 37.04 / -1.72 | yes | no | -0.19 | no |
| high_sales | 30.67 ± 1.22 (3) | -12.93 ± 1.25 (3) | 30.81 | 34.21 ± 2.74 (18) | 27.95 / -11.46 | yes | no | -0.99 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.947 | 89.84 | 1.592 | n/a | n/a |
| low_sales | 0.898 | 91.34 | 1.131 | n/a | n/a |
| high_sales | 1.081 | 85.54 | 2.915 | n/a | n/a |

## fc-tft-quantile (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 33.30 ± 0.31 (3) | -11.53 ± 2.95 (3) | 33.44 | 39.33 ± 2.82 (18) | 31.75 / -7.37 | yes | no | -0.55 | no |
| low_sales | 38.89 ± 1.02 (3) | -8.98 ± 8.07 (3) | 39.04 | 44.81 ± 3.06 (18) | 37.04 / -1.72 | yes | no | -0.60 | no |
| high_sales | 29.24 ± 0.62 (3) | -13.38 ± 1.23 (3) | 29.39 | 35.01 ± 4.15 (18) | 27.95 / -11.46 | yes | no | -0.31 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.954 | 89.84 | 1.592 | n/a | n/a |
| low_sales | 0.917 | 91.34 | 1.131 | n/a | n/a |
| high_sales | 1.053 | 85.54 | 2.915 | n/a | n/a |

## Paired bootstrap against the baseline

Test week, all series; forecasts averaged over seeds first. Each draw resamples series with replacement, the
same draw for both models, and recomputes WAPE over all rows. Delta = model WAPE - baseline WAPE (negative
favours the model), with its 95% bootstrap interval.

| Model | Imputation run | Point | Baseline | Baseline imputation run | WAPE | Baseline WAPE | Delta | 95% CI | Series | Draws |
|---|---|---|---|---|---|---|---|---|---|---|
| fc-dlinear-plus | imp-none-censored | estimated_demand | fc-dlinear-baseline | imp-none-censored | 30.70 | 31.74 | -1.05 | [-1.13, -0.96] | 50000 | 1000 |
| fc-tft-quantile | imp-none-censored | estimated_demand | fc-dlinear-baseline | imp-none-censored | 33.27 | 31.74 | 1.53 | [1.36, 1.72] | 50000 | 1000 |
| fc-dlinear-plus | imp-none-censored | q500 | fc-dlinear-baseline | imp-none-censored | 30.70 | 31.74 | -1.05 | [-1.13, -0.96] | 50000 | 1000 |
| fc-tft-quantile | imp-none-censored | q500 | fc-dlinear-baseline | imp-none-censored | 32.36 | 31.74 | 0.62 | [0.44, 0.81] | 50000 | 1000 |
