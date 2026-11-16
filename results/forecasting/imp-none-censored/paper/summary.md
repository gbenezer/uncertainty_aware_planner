# Forecast evaluation: imp-none-censored, paper protocol

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
- Paper protocol: fold CV comes from the project-protocol folds of the same model and data.

## fc-dlinear-baseline (imp-none-censored, paper, point estimated_demand)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.51 ± 0.02 (3) | -5.46 ± 0.02 (3) | 31.54 | 38.06 ± 0.80 (18) | 31.56 / -4.89 | yes | no | 0.06 | no |
| low_sales | 39.48 ± 0.00 (3) | -1.92 ± 0.03 (3) | 39.73 | 45.35 ± 0.65 (18) | 39.78 / -1.03 | yes | no | 0.46 | no |
| high_sales | 25.73 ± 0.03 (3) | -8.03 ± 0.02 (3) | 25.68 | 32.30 ± 0.84 (18) | 25.68 / -7.63 | yes | no | -0.05 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.910 | n/a | n/a | n/a | n/a |
| low_sales | 0.902 | n/a | n/a | n/a | n/a |
| high_sales | 0.933 | n/a | n/a | n/a | n/a |

## fc-dlinear-baseline (imp-none-censored, paper, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.51 ± 0.02 (3) | -5.46 ± 0.02 (3) | 31.54 | 38.06 ± 0.80 (18) | 31.56 / -4.89 | yes | no | 0.06 | no |
| low_sales | 39.48 ± 0.00 (3) | -1.92 ± 0.03 (3) | 39.73 | 45.35 ± 0.65 (18) | 39.78 / -1.03 | yes | no | 0.46 | no |
| high_sales | 25.73 ± 0.03 (3) | -8.03 ± 0.02 (3) | 25.68 | 32.30 ± 0.84 (18) | 25.68 / -7.63 | yes | no | -0.05 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.910 | n/a | n/a | n/a | n/a |
| low_sales | 0.902 | n/a | n/a | n/a | n/a |
| high_sales | 0.933 | n/a | n/a | n/a | n/a |

## fc-dlinear-plus (imp-none-censored, paper, point estimated_demand)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 30.51 ± 0.00 (3) | -11.05 ± 0.05 (3) | 30.48 | 38.23 ± 2.26 (18) | 31.56 / -4.89 | yes | no | 0.47 | no |
| low_sales | 37.73 ± 0.01 (3) | -10.84 ± 0.06 (3) | 37.83 | 45.80 ± 2.62 (18) | 39.78 / -1.03 | yes | no | 0.78 | no |
| high_sales | 25.26 ± 0.01 (3) | -11.21 ± 0.04 (3) | 25.21 | 32.26 ± 1.97 (18) | 25.68 / -7.63 | yes | no | 0.21 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.886 | n/a | n/a | n/a | n/a |
| low_sales | 0.871 | n/a | n/a | n/a | n/a |
| high_sales | 0.927 | n/a | n/a | n/a | n/a |

## fc-dlinear-plus (imp-none-censored, paper, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 30.51 ± 0.00 (3) | -11.05 ± 0.05 (3) | 30.48 | 38.23 ± 2.26 (18) | 31.56 / -4.89 | yes | no | 0.47 | no |
| low_sales | 37.73 ± 0.01 (3) | -10.84 ± 0.06 (3) | 37.83 | 45.80 ± 2.62 (18) | 39.78 / -1.03 | yes | no | 0.78 | no |
| high_sales | 25.26 ± 0.01 (3) | -11.21 ± 0.04 (3) | 25.21 | 32.26 ± 1.97 (18) | 25.68 / -7.63 | yes | no | 0.21 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.886 | n/a | n/a | n/a | n/a |
| low_sales | 0.871 | n/a | n/a | n/a | n/a |
| high_sales | 0.927 | n/a | n/a | n/a | n/a |

## fc-tft-quantile (imp-none-censored, paper, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.63 (1) | -7.40 (1) | 31.73 | 39.33 ± 2.82 (18) | 31.75 / -7.37 | yes | no | 0.04 | no |
| low_sales | 37.00 (1) | -5.57 (1) | 37.15 | 44.81 ± 3.06 (18) | 37.04 / -1.72 | yes | no | 0.01 | no |
| high_sales | 27.73 (1) | -8.73 (1) | 27.83 | 35.01 ± 4.15 (18) | 27.95 / -11.46 | yes | no | 0.05 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.877 | 91.72 | 1.587 | n/a | n/a |
| low_sales | 0.852 | 92.56 | 1.129 | n/a | n/a |
| high_sales | 0.944 | 89.32 | 2.903 | n/a | n/a |

## Paired bootstrap against the baseline

Test week, all series; forecasts averaged over seeds first. Each draw resamples series with replacement, the
same draw for both models, and recomputes WAPE over all rows. Delta = model WAPE - baseline WAPE (negative
favours the model), with its 95% bootstrap interval.

| Model | Imputation run | Point | Baseline | Baseline imputation run | WAPE | Baseline WAPE | Delta | 95% CI | Series | Draws |
|---|---|---|---|---|---|---|---|---|---|---|
| fc-dlinear-plus | imp-none-censored | estimated_demand | fc-dlinear-baseline | imp-none-censored | 30.51 | 31.51 | -1.01 | [-1.10, -0.92] | 50000 | 1000 |
| fc-dlinear-plus | imp-none-censored | q500 | fc-dlinear-baseline | imp-none-censored | 30.51 | 31.51 | -1.01 | [-1.10, -0.92] | 50000 | 1000 |
| fc-tft-quantile | imp-none-censored | q500 | fc-dlinear-baseline | imp-none-censored | 31.63 | 31.51 | 0.12 | [-0.12, 0.36] | 50000 | 1000 |
