# DLinear forecasting baseline (`fc-dlinear-baseline`) and the improved DLinear (`fc-dlinear-plus`)

**Owner:** Vishnu. **Data:** FreshRetailNet-50K, censored setting (`imp-none-censored`).

**Numbers:**
- `dlinear_paper_check_wape.csv`: the reproduction.
- `imp-none-censored/{project,paper}/`: `summary.md`, `metrics.csv` (CONTRACTS.md 2.10) and `paper_wape.csv`.
- `dlinear_ablation*.{csv,md}`: validation-only selection.
- `test_ledger.csv`: every test-week scoring.

**How WAPE is computed (after review):**
- **Official WAPE** follows the literature (Dev Contract A.4): Σ|F − A| / Σ|A| over all scored rows. Each row is one store-product-day, so no errors are netted before the absolute value.
- **Paper-style WAPE** is the dataset paper's own scoring: the same ratio per test day, then averaged over the 7 days. It is shown only to compare with the published numbers, which use it.
- **Rows scored:** stockout-free days only, against raw sales. The two definitions differ by at most 0.25 points here.

**TL;DR.**
- **The port reproduces the paper.** On the paper's training split our DLinear scores **31.54% ± 0.01** paper-style (official **31.51%**), against the published **31.56%** (raw sales, paper Table 3). The psd slices match: 25.68% for psd ≥ 1 (published 25.68%) and 39.73% for psd < 1 (published 39.78%).
- **Project split** (calibration week held out): **31.74% ± 0.01**. It passes Dev Contract A.7.2 against 31.56% + SD_cv, with SD_cv = **0.80** (6 folds × 3 seeds).
- **It beats our TFT.** With seed-averaged forecasts, the DLinear is 0.62 points better than the TFT median (95% bootstrap CI [0.44, 0.81]).
- **The improved DLinear (`fc-dlinear-plus`)** scores **30.70% ± 0.02** on the project split and **30.51%** on the paper split.
  - Its changes were chosen on validation data only.
  - That is about 1.0 point better than the baseline (CI excludes 0) and than the published 31.56%.
  - It is ≥ 1 SD better against the baseline's SD_cv, but not against its own (inflated) fold SD; see below.
  - Trade-off: it under-forecasts more (WPE about −11%).

## What the baseline is

Feature Cards 2.1: "DLinear architecture trained using the loss and exact architecture from the dataset paper repository (not quantile)". `models/forecasting/dlinear.py` is a port of `Dingdong-Inc/frn-50k-baseline @ 8af694a`, `demand_forecasting/DLinear` (Apache-2.0), with the run settings of their `train_predict.sh`.

| Area | Setting |
|---|---|
| Model | moving-average decomposition (kernel 28; even, so 13 days padded in front and 14 at the end), one shared `Linear(62 → 7)` for the seasonal part and one for the trend |
| Normalisation | one StandardScaler over every series' training days, then RevIN per window (no affine weights) |
| Training | L1 loss, Adam lr 1e-3 (1e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5 over 6 epochs), batch 1024, drop_last, best training-loss epoch kept, no validation set |
| Windows | 62-day lookback, 7-day horizon, stride 1 inside the training window (15 per series on the project split, 22 on the paper split) |
| Output | one forecast, the median of an L1 fit. It is written as `q500` and `estimated_demand`; `q025`/`q975` stay null (CONTRACTS.md 2.5, non-quantile baseline). Negative forecasts are clipped to 0. |

**Two details of the authors' code that had to be read carefully:**
1. **RevIN is on.** `train_predict.sh` passes `--revin False`, but `main.py` parses it with `argparse type=bool`, and `bool("False")` is `True`.
2. **The learning rate is halved only from the third epoch.** `adjust_learning_rate(epoch + 1)` runs after each epoch, so epochs 1 and 2 both use 1e-3.

Their model takes 7 channels (6 covariates and the target), but its channel-shared linear layers never mix channels and the loss reads only the target. **The covariates therefore cannot change the forecast**: the published DLinear is univariate. `tests/models/forecasting/test_dlinear_baseline.py` checks this against their code, and checks that our port gives their outputs for the same weights.

**Data path.** `models/forecasting/dense_windows.py` holds the daily series as `[series, day]` arrays: about 0.1 GB against about 13 GB for the pytorch-forecasting datasets. The tests check that its training and evaluation windows equal those of `data/timeseries_dataset.py` for all 6 folds and both final parts, and that test-week targets never reach the model. A full run of 6 folds + final takes about 5 minutes on 8 CPU cores.

## Results (test week 06-26..07-02, raw sales, stockout-free days, 3 seeds)

| Run | Train | WAPE all | Paper-style | WPE all | WAPE psd ≥ 1 | WAPE psd < 1 | MASE |
|---|---|---|---|---|---|---|---|
| DLinear, published (raw sales) | 03-28..06-25 | n/a | **31.56%** | −4.89% | 25.68% | 39.78% | n/a |
| **Baseline, paper split** | 03-28..06-25 | **31.51 ± 0.02%** | 31.54% | −5.46% | 25.73% | 39.48% | 0.910 |
| **Plus, paper split** | 03-28..06-25 | **30.51 ± 0.00%** | 30.48% | −11.05% | 25.26% | 37.73% | 0.886 |
| TFT median, paper split (1 seed) | 03-28..06-25 | 31.63% | 31.73% | −7.40% | 27.73% | 37.00% | 0.877 |
| **Baseline, project split** | 03-28..06-18 | **31.74 ± 0.01%** | 31.76% | −7.38% | 26.19% | 39.38% | 0.933 |
| **Plus, project split** | 03-28..06-18 | **30.70 ± 0.02%** | 30.67% | −11.89% | 25.49% | 37.87% | 0.911 |
| TFT median, project split | 03-28..06-18 | 33.30 ± 0.31% | 33.44% | −11.53% | 29.24% | 38.89% | 0.954 |

**Paired bootstrap** (seed-averaged forecasts, series resampled, 1000 draws):

| Comparison | Project split | Paper split |
|---|---|---|
| Plus − baseline | **−1.05 [−1.13, −0.96]** | **−1.01 [−1.10, −0.92]** |
| TFT median − baseline | +0.62 [0.44, 0.81] | +0.12 [−0.12, 0.36] |

**Project-split extras:**
- **Fold CV:** baseline 38.06% ± 0.80, plus 38.23% ± 2.26, TFT median 39.33% ± 2.82 (6 folds × 3 seeds).
- **Holding out the calibration week** costs the baseline 0.23 points (31.51% → 31.74%), against about 1.7 points for the TFT.

**Contract A.7.2 for the baseline (SD_cv = 0.80):**

| Check | Result |
|---|---|
| Pass: 31.74 ≤ 31.56 + 0.80 | ✅ |
| Literal: \|fold mean − published\| ≤ SD_cv | ❌ (6.5 points) |

The literal check fails for every model, ours and the TFT alike, because the folds are much harder than the test week:
- Their validation windows include the May Day holidays.
- A fold trains a 21-day lookback on 28 days of data: one window per series.

## Improved DLinear (`fc-dlinear-plus`)

**Changes.** All of these are flags on the same model (`configs/forecasting_models/dlinear_forecasting_plus.yaml`) and were chosen on validation data only:
- **Lookback 28 days**, giving 49 training windows per series instead of 15 on the project split.
- **The L1 loss skips target days with a daytime stockout**, whose recorded sales understate demand.
- **RevIN takes its level from in-stock lookback days only.**
- **A linear head over the planned inputs of the 7 forecast days**: discount, activity_flag, holiday_flag and day of week. The published DLinear never uses its covariates. Observed future weather is deliberately left out, since a deployed model would only have a forecast.

**Selection (`dlinear_ablation.md`, `dlinear_ablation_pairs.md`).**
- **Data used:** 11 variants × 3 seeds, scored on the inner week 06-12..06-18 (trained through 06-11) and on the 6 team folds. Calib and test were never used.
- **Choice:** the frozen config has the best inner-week WAPE, **31.72% against 34.22%** for the baseline (paper-style scoring, paired, 3 of 3 seeds better).
- **Why the folds don't decide it:** the masked loss with in-stock RevIN also improves all 18 fold runs. The planned-input head cannot be learned on folds, because a fold trains on one window per series, all with the same dates.
- **Test-week scoring:** done once per protocol for the frozen config. `test_ledger.csv` records the table hashes; later rows re-score the same files (the hashes match).

**Beating the published WAPE by ≥ 1 SD (an optional team goal, not a contract term): it depends on which SD.**

| SD used | Paper split, 31.56 − 30.51 | Project split, 31.56 − 30.70 | Met? |
|---|---|---|---|
| Baseline's fold SD (0.80) | 1.31 SD | 1.08 SD | ✅ |
| Plus's own fold SD (2.26) | 0.46 SD | 0.38 SD | ❌ |

- The plus model's own SD is inflated by folds where its planned-input head cannot be learned.
- `summary.md` applies each model's own SD.

**Caveat on bias.**
- Plus under-forecasts more than the baseline: WPE about −11% against −5% to −7%. The published model is at −4.89%.
- An L1 loss targets the median, and on right-skewed demand the median sits below the mean. The masked loss also trains on in-stock days.
- WAPE improves, and so does slow-seller accuracy: low sellers are 2 points better than published.
- Conformal calibration on the calibration week, or a level correction fitted there, is the natural next step. Neither may be tuned on the test week.

## Reproduce (Explorer, CPU)
```bash
# reproduction on the paper's split, then the project grid (6 folds + final) and the evaluation, one command each
python experiments/vishnu/paper_split_run.py --config configs/forecasting_models/dlinear_forecasting_baseline.yaml \
    --demand-daily data/processed/demand_daily/imp-none-censored.parquet --seed 2025
sbatch slurm/run_experiment.sbatch configs/experiments/dlinear_baseline_censored.yaml
sbatch slurm/run_experiment.sbatch configs/experiments/dlinear_baseline_censored_paper.yaml
# improved model, validation-only selection, then its runs
sbatch slurm/cpu_job.sbatch python experiments/vishnu/dlinear_ablation.py \
    --demand-daily data/processed/demand_daily/imp-none-censored.parquet --out results/forecasting
sbatch slurm/run_experiment.sbatch configs/experiments/dlinear_plus_censored.yaml
sbatch slurm/run_experiment.sbatch configs/experiments/dlinear_plus_censored_paper.yaml
```
