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

## fc-dlinear-quantile (imp-none-censored, project, point q500)

| Slice | Test WAPE | Test WPE | Paper-style WAPE | Fold CV WAPE | Published WAPE / WPE | Pass | >= 1 SD better (optional) | Improvement (SD_cv) | A.7.2 literal |
|---|---|---|---|---|---|---|---|---|---|
| all | 31.91 (1) | -2.54 (1) | 31.92 | n/a | 31.56 / -4.89 | n/a | n/a | n/a | n/a |
| low_sales | 39.75 (1) | 0.95 (1) | 39.99 | n/a | 39.78 / -1.03 | n/a | n/a | n/a | n/a |
| high_sales | 26.21 (1) | -5.07 (1) | 26.14 | n/a | 25.68 / -7.63 | n/a | n/a | n/a | n/a |

| Slice | Test MASE | Test PICP95 | Test MPIW95 | Test PICP95 (cal) | Test MPIW95 (cal) |
|---|---|---|---|---|---|
| all | 0.936 | 95.85 | 2.097 | n/a | n/a |
| low_sales | 0.924 | 96.87 | 1.527 | n/a | n/a |
| high_sales | 0.969 | 92.90 | 3.737 | n/a | n/a |
