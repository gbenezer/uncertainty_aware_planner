# DLinear ablation on validation data (inner week 06-12..06-18 and the 6 team folds)

Raw-sales truth, stockout-free days, paper-style WAPE. Lower is better. Never uses calib or test.
Selection uses the inner week (8 training origins per series at lookback 62). A fold trains on one
window per series, all with the same dates, so calendar inputs (day of week, holidays) cannot be
learned there: folds are a secondary check and the source of the contract's SD.

| variant | inner-week WAPE (mean ± SD over seeds) | inner-week WPE | inner-week ΔWAPE vs baseline (paired by seed) | fold WAPE (mean ± SD) | fold ΔWAPE vs baseline (paired) |
|---|---|---|---|---|---|
| baseline | 34.22% ± 0.01% | -4.09% | +0.00% (0/3 better) | 38.24% ± 0.78% | +0.00% (0/18 better) |
| mask+instock_revin | 33.11% ± 0.02% | -7.29% | -1.11% (3/3 better) | 37.07% ± 0.75% | -1.17% (18/18 better) |
| instock_revin+planned | 33.32% ± 0.06% | -4.98% | -0.90% (3/3 better) | 39.31% ± 2.35% | +1.07% (6/18 better) |
