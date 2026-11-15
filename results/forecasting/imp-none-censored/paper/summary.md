# Forecast evaluation: imp-none-censored, paper protocol

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
- Paper protocol: fold CV comes from the project-protocol folds of the same model and data.

## fc-dlinear-baseline (imp-none-censored, paper, point estimated_demand)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.54 ± 0.01 (3) | -5.16 ± 0.02 (3) | 31.51 | 38.24 ± 0.78 (18) | 31.56 / -4.89 | yes | no | 0.03 | no |
| low_sales | 39.73 ± 0.00 (3) | -1.27 ± 0.02 (3) | 39.48 | 45.70 ± 0.67 (18) | 39.78 / -1.03 | yes | no | 0.08 | no |
| high_sales | 25.68 ± 0.02 (3) | -7.93 ± 0.02 (3) | 25.73 | 32.38 ± 0.80 (18) | 25.68 / -7.63 | yes | no | 0.00 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.910 | n/a | n/a | n/a | n/a |
| low_sales | 0.902 | n/a | n/a | n/a | n/a |
| high_sales | 0.933 | n/a | n/a | n/a | n/a |

## fc-dlinear-baseline (imp-none-censored, paper, point q500)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.54 ± 0.01 (3) | -5.16 ± 0.02 (3) | 31.51 | 38.24 ± 0.78 (18) | 31.56 / -4.89 | yes | no | 0.03 | no |
| low_sales | 39.73 ± 0.00 (3) | -1.27 ± 0.02 (3) | 39.48 | 45.70 ± 0.67 (18) | 39.78 / -1.03 | yes | no | 0.08 | no |
| high_sales | 25.68 ± 0.02 (3) | -7.93 ± 0.02 (3) | 25.73 | 32.38 ± 0.80 (18) | 25.68 / -7.63 | yes | no | 0.00 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.910 | n/a | n/a | n/a | n/a |
| low_sales | 0.902 | n/a | n/a | n/a | n/a |
| high_sales | 0.933 | n/a | n/a | n/a | n/a |

## fc-dlinear-plus (imp-none-censored, paper, point estimated_demand)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 30.48 ± 0.00 (3) | -11.03 ± 0.05 (3) | 30.51 | 38.51 ± 2.48 (18) | 31.56 / -4.89 | yes | no | 0.43 | no |
| low_sales | 37.83 ± 0.01 (3) | -10.82 ± 0.06 (3) | 37.73 | 46.23 ± 2.89 (18) | 39.78 / -1.03 | yes | no | 0.67 | no |
| high_sales | 25.21 ± 0.01 (3) | -11.21 ± 0.04 (3) | 25.26 | 32.43 ± 2.15 (18) | 25.68 / -7.63 | yes | no | 0.22 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.886 | n/a | n/a | n/a | n/a |
| low_sales | 0.871 | n/a | n/a | n/a | n/a |
| high_sales | 0.927 | n/a | n/a | n/a | n/a |

## fc-tft-quantile (imp-none-censored, paper, point q500)

| Slice | Test WAPE | Test WPE | Pooled WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | Stretch | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.73 (1) | -7.76 (1) | 31.63 | 39.48 ± 2.85 (18) | 31.75 / -7.37 | yes | no | 0.01 | no |
| low_sales | 37.15 (1) | -5.79 (1) | 37.00 | 45.12 ± 3.27 (18) | 37.04 / -1.72 | yes | no | -0.03 | no |
| high_sales | 27.83 (1) | -9.15 (1) | 27.73 | 35.06 ± 4.11 (18) | 27.95 / -11.46 | yes | no | 0.03 | no |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.877 | 91.72 | 1.587 | n/a | n/a |
| low_sales | 0.852 | 92.56 | 1.129 | n/a | n/a |
| high_sales | 0.944 | 89.32 | 2.903 | n/a | n/a |

## Paired bootstrap against the baseline

Test week, all series; forecasts averaged over seeds first. Each draw resamples series with replacement, the
same draw for both models. Delta = model WAPE - baseline WAPE (negative favours the model), with its 95%
bootstrap interval.

| Model | Imputation run | Point | Baseline | Baseline imputation run | WAPE | Baseline WAPE | Delta | 95% CI | Series | Draws |
|---|---|---|---|---|---|---|---|---|---|---|
| fc-tft-quantile | imp-none-censored | q500 | fc-dlinear-baseline | imp-none-censored | 31.73 | 31.54 | 0.20 | [-0.04, 0.43] | 50000 | 1000 |
| fc-dlinear-plus | imp-none-censored | estimated_demand | fc-dlinear-baseline | imp-none-censored | 30.48 | 31.54 | -1.05 | [-1.15, -0.97] | 50000 | 1000 |
