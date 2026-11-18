# DLinear ablation on validation data (inner week 06-12..06-18 and the 6 team folds)

Raw-sales truth, stockout-free days, paper-style WAPE. Lower is better. Never uses calib or test.
Selection uses the inner week (8 training origins per series at lookback 62). A fold trains on one
window per series, all with the same dates, so calendar inputs (day of week, holidays) cannot be
learned there: folds are a secondary check and the source of the contract's SD.

| variant | inner-week WAPE (mean ± SD over seeds) | inner-week WPE | inner-week ΔWAPE vs baseline (paired by seed) | fold WAPE (mean ± SD) | fold ΔWAPE vs baseline (paired) |
|---|---|---|---|---|---|
| baseline | 34.22% ± 0.01% | -4.09% | +0.00% (0/3 better) | 38.24% ± 0.78% | +0.00% (0/18 better) |
| mask | 34.14% ± 0.01% | -5.79% | -0.08% (3/3 better) | 38.14% ± 0.78% | -0.10% (14/18 better) |
| instock_revin | 33.83% ± 0.02% | -4.76% | -0.39% (3/3 better) | 37.89% ± 0.80% | -0.35% (18/18 better) |
| planned | 33.86% ± 0.06% | -4.22% | -0.35% (3/3 better) | 39.78% ± 2.36% | +1.53% (6/18 better) |
| planned_no_dow | 33.91% ± 0.07% | -4.10% | -0.31% (3/3 better) | 39.53% ± 1.86% | +1.28% (6/18 better) |
| mask+planned | 33.80% ± 0.05% | -5.86% | -0.42% (3/3 better) | 39.83% ± 2.46% | +1.58% (5/18 better) |
| all | 32.63% ± 0.04% | -7.19% | -1.59% (3/3 better) | 38.51% ± 2.48% | +0.27% (10/18 better) |
| lookback28 | 33.94% ± 0.02% | -1.53% | -0.28% (3/3 better) | 38.24% ± 0.78% | +0.00% (0/18 better) |
| lookback28+all | 31.72% ± 0.02% | -4.52% | -2.50% (3/3 better) | 38.51% ± 2.48% | +0.27% (10/18 better) |
