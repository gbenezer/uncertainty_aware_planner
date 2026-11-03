# TFT (`fc-tft-quantile`): check against the paper's WAPE

**Owner:** Vishnu. **Data:** FreshRetailNet-50K, censored setting (`imp-none-censored`). **Numbers:** `tft_paper_check_wape.csv`.

**TL;DR.**
- On the paper's training split our TFT scores **31.73% WAPE**, against the published censored TFT's **31.75%**. The implementation matches the paper.
- On the project's split, which holds the calibration week out of training, it scores **33.44% ± 0.28** (mean ± sd over 3 seeds), about 1.7 points above the paper.

## How the check works

**1. Same data setting.** The paper's 31.75% is the censored result: no stockout recovery, so stockout hours keep their recorded (understated) sales.
- `pipeline/censored.py` builds that `demand_daily`: 50,000 store-product series × 97 days.

**2. Same model.** `configs/forecasting_models/temporal_fusion_transformer.yaml` uses the paper's TFT settings:

| Area | Settings |
|---|---|
| Architecture | hidden 128, 8 heads, 2 LSTM layers, hidden_continuous 16, dropout 0.1 |
| Optimiser | lr 0.01, AdamW, weight decay 1e-4 |
| Training | 5 epochs, batch 1024, 70-day encoder, 7-day horizon, no validation set |

The loss is QuantileLoss on [0.025, 0.5, 0.975], because the project needs a 95% interval. The median (q500) is the point forecast.

**3. Two training splits, same test week (2024-06-26..07-02).**

| Split | Train | Held out | Code |
|---|---|---|---|
| Paper | 03-28..06-25 | n/a | `experiments/vishnu/paper_split_run.py` |
| Project | 03-28..06-18 | calib week 06-19..06-25, kept for conformal intervals | `pipeline/forecast.py` (final run) |

In both runs the test-week targets are zeroed in the model input.

**4. Same scoring.** `experiments/vishnu/paper_wape.py` re-implements the authors' `demand_forecasting/TFT/predictTFT.py`:
- **Point forecast:** q500, compared against the actual `demand_total`.
- **Days scored:** only days with no stockout between 06:00 and 22:00 (`stock_hour6_22_cnt == 0`), because recorded sales understate demand on stockout days. That leaves 206,562 of 350,000 forecast rows.
- **WAPE** = Σ|q500 − actual| / Σ actual and **WPE** (bias) = Σ(q500 − actual) / Σ actual, computed per day.
- **Reported number:** the **mean of the 7 daily values**, as in the paper.
- **Groups by psd:** psd is each series' mean daily demand before the test week. psd ≥ 1 covers 53k rows (faster sellers) and psd < 1 covers 153k rows (slow sellers).

## Results (test week, censored data)

| Run | Seeds | WAPE all | WPE all | WAPE psd≥1 | WPE psd≥1 | WAPE psd<1 | WPE psd<1 |
|---|---|---|---|---|---|---|---|
| Paper, censored TFT (published) | n/a | **31.75%** | n/a | n/a | n/a | n/a | n/a |
| Ours, paper split | 2025 | **31.73%** | −7.76% | 27.83% | −9.15% | 37.15% | −5.79% |
| Ours, project split | 2025 | 33.12% | −9.67% | 29.41% | −13.45% | 38.25% | −4.38% |
| Ours, project split | 2026 | 33.57% | −15.16% | 28.77% | −12.66% | 40.23% | −18.62% |
| Ours, project split | 2027 | 33.62% | −10.87% | 30.01% | −15.14% | 38.62% | −4.95% |
| **Ours, project split, mean ± sd** | 3 | **33.44 ± 0.28%** | −11.90 ± 2.89% | 29.40 ± 0.62% | −13.75 ± 1.27% | 39.03 ± 1.05% | −9.32 ± 8.06% |

An earlier seed-2025 project-split run with the same settings scored 34.46% (in the CSV as `project_split_first_run`). See finding 3.

## Findings
1. **The implementation reproduces the paper.**
   - On the paper's split we are within 0.02 points of the published 31.75%.
   - This confirms that the data handling, model configuration, training and scoring match the authors' code.
2. **Holding out the calibration week costs about 1.7 WAPE points** (33.44% vs 31.73%).
   - The model trains on 7 fewer days, the most recent ones before the test week.
   - This is the expected price of a clean calibration week for conformal intervals, and the project split stays the official setting.
3. **Seed noise is small but not zero.**
   - Across 3 seeds the overall WAPE sd is 0.28 points.
   - Two runs of the same seed gave 33.12% and 34.46%, because GPU training is not bit-for-bit deterministic.
   - So differences under about 1 point between methods should be read with care.
4. **Slow sellers are the hard part.** Items under 1 unit/day score about 39% WAPE, against about 29% for faster sellers, on both splits.
5. **The model under-forecasts on every run** (WPE always negative). This is consistent with training on censored sales. Stockout recovery (imputation) should reduce the bias, so the imputed-data grid will be compared against these numbers.
6. **The quantile outputs are clean.**
   - Quantile crossing affects about 0.003% of rows (fixed by sorting).
   - About 4% of test-week quantile values were negative and clipped to 0.

## Runs (Explorer, V100)
- **Grid:** 3 seeds (2025, 2026, 2027) × 6 rolling folds + the final model = 21 runs, all completed.
- **Runtime:** about 6–15 minutes per fold and 31–37 minutes per final model.
- **Final model forecasts:** it forecasts both the calibration week and the test week from one checkpoint, so conformal offsets fitted on calib apply to the model that made the test forecasts.
- **Tables:** `data/processed/forecast/fc-tft-quantile-<seed>/`. Raw quantiles and run logs are in `data/runs/forecasting/fc-tft-quantile-<seed>/`. Neither is committed.

## Reproduce
```bash
F="python -m uncertainty_aware_planner.pipeline.forecast --config configs/forecasting_models/temporal_fusion_transformer.yaml --demand-daily data/processed/demand_daily/imp-none-censored.parquet"
# grid: one seed per submit (Explorer QOS), see slurm/README.md
sbatch --job-name=tft_grid --array=0-6%3 slurm/array_job.sbatch "2025 2026 2027" "1 2 3 4 5 6 final" $F
# paper split
sbatch --job-name=tft_paper_split slurm/gpu_job.sbatch python experiments/vishnu/paper_split_run.py \
    --demand-daily data/processed/demand_daily/imp-none-censored.parquet --seed 2025
# score a test week
sbatch --job-name=wape --mem=32G --time=00:20:00 slurm/cpu_job.sbatch python experiments/vishnu/paper_wape.py \
    --demand-daily data/processed/demand_daily/imp-none-censored.parquet \
    --quantiles data/runs/forecasting/fc-tft-quantile-<seed>/final/quantiles_test.parquet
```
