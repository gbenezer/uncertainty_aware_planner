# DLinear forecasting baseline (`fc-dlinear-baseline`) and the improved DLinear (`fc-dlinear-plus`)

**Owner:** Vishnu. **Data:** FreshRetailNet-50K, censored setting (`imp-none-censored`).

**Numbers:**
- `dlinear_paper_check_wape.csv` (reproduction)
- `imp-none-censored/{project,paper}/summary.md` and `metrics.csv` (evaluation, CONTRACTS.md 2.10)
- `dlinear_ablation.{csv,md}` (validation-only selection)
- `test_ledger.csv` (every test-week scoring)

**TL;DR.**
- **The port reproduces the paper.**
  - On the paper's training split our DLinear scores **31.54% ± 0.01** WAPE (3 seeds), against the published **31.56%** (raw sales, paper Table 3).
  - The slices also match: psd ≥ 1 is 25.68% (published 25.68%) and psd < 1 is 39.73% (published 39.78%).
- **On the project split** (calibration week held out) it scores **31.76% ± 0.01**. That **passes** contract A.7.2 against 31.56% ± SD_cv, where SD_cv = **0.78** (6 folds × 3 seeds).
- **It beats our TFT.** On the same split, with seed-averaged forecasts, the DLinear is 0.73 points better than the TFT median (95% bootstrap CI [0.55, 0.92]).
- **The improved DLinear (`fc-dlinear-plus`)** scores **30.67% ± 0.02** on the project split and **30.48%** on the paper split.
  - Its changes were chosen on validation data only.
  - That is about 1.1 points better than the baseline (CI excludes 0) and than the published 31.56%.
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

## Results (test week 06-26..07-02, raw sales, stockout-free days, paper scoring)

| Run | Train | Seeds | WAPE all | WPE all | WAPE psd ≥ 1 | WAPE psd < 1 |
|---|---|---|---|---|---|---|
| DLinear, published (raw sales) | 03-28..06-25 | n/a | **31.56%** | −4.89% | 25.68% | 39.78% |
| **Ours, paper split** | 03-28..06-25 | 3 | **31.54 ± 0.01%** | −5.17% | 25.68% | 39.73% |
| **Ours, project split** | 03-28..06-18 | 3 | **31.76 ± 0.01%** | −7.05% | 26.14% | 39.60% |
| TFT median, project split (for comparison) | 03-28..06-18 | 3 | 33.44 ± 0.27% | −11.90% | 29.39% | 39.04% |

**Project-split extras:**
- **Fold CV:** 38.24% ± 0.78 (6 folds × 3 seeds).
- **Test MASE:** 0.933.
- **Holding out the calibration week costs 0.22 points:** 31.54% → 31.76%, against about 1.7 points for the TFT.

**Contract A.7.2 (SD_cv = 0.78):**

| Check | Result |
|---|---|
| Pass: 31.76 ≤ 31.56 + 0.78 | ✅ |
| Stretch: 31.76 ≤ 31.56 − 0.78 = 30.78 | ❌ |
| Literal: \|fold mean − published\| ≤ SD_cv | ❌ (6.68 points) |

The literal check fails for every model, ours and the TFT alike, because the folds are much harder than the test week:
- Their validation windows include the May Day holidays.
- A fold trains a 21-day lookback on 28 days of data: one window per series.

## Findings
1. **The port reproduces the paper.** It is within 0.02 points overall, and the slices are within 0.05.
2. **Seed noise is tiny for a linear model.** The SD is about 0.01 points, so the contract's SD has to come from the folds.
3. **The model under-forecasts** (WPE −5% to −7%), as expected from training on censored sales. Days with a stockout are the high-demand days, so their recorded sales pull the fit down.
4. **Folds cannot judge calendar inputs.** A fold trains on one window per series, all with the same dates, so day-of-week and holiday effects are not identifiable. Variants with planned inputs therefore look worse on folds than they are. Model selection for the improved DLinear uses an inner validation week, 06-12..06-18, trained through 06-11 with 8 origins per series; the folds are a secondary check. See `dlinear_ablation.md`.

## Improved DLinear (`fc-dlinear-plus`)

**Changes.** All of these are flags on the same model (`configs/forecasting_models/dlinear_forecasting_plus.yaml`) and were chosen on validation data only:
- **Lookback 28 days**, giving 49 training windows per series instead of 15 on the project split.
- **The L1 loss skips target days with a daytime stockout**, whose recorded sales understate demand.
- **RevIN takes its level from in-stock lookback days only.**
- **A linear head over the planned inputs of the 7 forecast days**: discount, activity_flag, holiday_flag and day of week. The published DLinear never uses its covariates. Observed future weather is deliberately left out, since a deployed model would only have a forecast.

**Selection (`dlinear_ablation.md`, `dlinear_ablation_pairs.md`).**
- **Data used:** 11 variants × 3 seeds, scored on the inner week 06-12..06-18 (trained through 06-11) and on the 6 team folds. Calib and test were never used.
- **Choice:** the frozen config has the best inner-week WAPE, **31.72% against 34.22%** for the baseline (paired, 3 of 3 seeds better).
- **Why the folds don't decide it:** the masked loss with in-stock RevIN also improves all 18 fold runs (−1.17 points). The planned-input head cannot be learned on folds (see finding 4).
- **Test-week scoring:** done once per protocol for the frozen config. `test_ledger.csv` records the table hashes; the later rows re-score the same files.

**Results (test week, raw sales, 3 seeds).**

| Run | WAPE all | WPE all | WAPE psd ≥ 1 | WAPE psd < 1 | MASE | Δ vs baseline (95% bootstrap CI) |
|---|---|---|---|---|---|---|
| Published DLinear (raw sales) | 31.56% | −4.89% | 25.68% | 39.78% | n/a | |
| Baseline, paper split | 31.54 ± 0.01% | −5.16% | 25.68% | 39.73% | | |
| **Plus, paper split** | **30.48 ± 0.00%** | −11.03% | 25.21% | 37.83% | | **−1.05 [−1.15, −0.97]** |
| Baseline, project split | 31.76 ± 0.01% | −7.05% | 26.14% | 39.60% | 0.933 | |
| **Plus, project split** | **30.67 ± 0.02%** | −11.84% | 25.43% | 37.97% | 0.911 | **−1.08 [−1.17, −0.99]** |

**Stretch goal (beat the published WAPE by ≥ 1 SD_cv): it depends on which SD.**

| SD used | Paper split, 31.56 − 30.48 | Project split, 31.56 − 30.67 | Met? |
|---|---|---|---|
| Baseline's fold SD (0.78) | 1.38 SD | 1.14 SD | ✅ |
| Plus's own fold SD (2.48) | 0.43 SD | 0.36 SD | ❌ |

- The plus model's own SD comes from folds where its planned-input head cannot be learned, so it is inflated.
- `summary.md` applies each model's own SD, so it reports stretch = no.
- **The team needs to choose which SD the contract claim uses.** Either way, the model passes A.7.2.
- **Our recommendation:** use the reference model's SD and state both numbers.

**Caveat on bias.**
- Plus under-forecasts more than the baseline: WPE about −11% against −5% to −7%. The published model is at −4.89%.
- An L1 loss targets the median, and on right-skewed demand the median sits below the mean. The masked loss also trains on in-stock days, whose recorded sales are lower.
- WAPE and slow-seller accuracy improve: low sellers are 1.8 points better than published. Planners who care about stockouts should still see the bias.
- Conformal calibration on the calib week (Tarakaram) or a level correction fitted there are the natural next steps. Neither may be tuned on the test week.

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
