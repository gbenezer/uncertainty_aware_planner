# Uncertainty-Aware Demand and Inventory Planning Tool

CS 6170 AI Capstone, Fall 2026, Northeastern University (Khoury College of Computer Sciences)

**Team:** Gil Benezer, Jeena Shrestha, Tarakaram Yadalla, Vishnu Purohitham
**Originator:** Pratyush Tyagi

A research prototype that recovers demand hidden by stockouts, forecasts store-product demand 7 days ahead, and attaches a calibrated uncertainty range to every estimate. People make the stocking decisions; the system only forecasts and shows how reliable each forecast is.

> **Scope reminder.** This is an offline research prototype on public data. There is no live ERP or ordering connection, no stock, safety-stock, or reorder recommendations, and no pricing decisions. See [Out of scope](#out-of-scope).

---

## Table of contents

1. [What this builds](#what-this-builds)
2. [Repository layout](#repository-layout)
3. [Quick start](#quick-start)
4. [Reproducing the baseline](#reproducing-the-baseline)
5. [Running the pipeline](#running-the-pipeline)
6. [Data](#data)
7. [Experiment design](#experiment-design)
8. [Ownership](#ownership)
9. [Working agreements](#working-agreements)
10. [Out of scope](#out-of-scope)
11. [Attribution](#attribution)

---

## What this builds

Two AI features, chained so that Feature 2 learns from the demand Feature 1 recovers.

| Feature | Input | Output | Models |
|---|---|---|---|
| **1. Demand imputation** | Hourly sales, hourly stockout flags, covariates (discount, holiday, activity, weather) | Estimated demand for every sold-out hour, with calibrated 95% confidence interval | DLinear or TimesNet (deployment decision TBD), 0.025 and 0.975 quantile outputs, conformalized |
| **2. 7-day forecast** | Daily demand (observed plus recovered), past and planned future covariates | Daily forecast per store-product with calibrated 95% confidence interval; review flag when the range is unusually wide or recent demand keeps landing outside it | DLinear or TFT(deployment decision TBD), 0.025 and 0.975 quantile outputs, conformalized |

Uncertainty ranges come from quantile (pinball-loss) models at the 2.5% and 97.5% levels, then calibrated with conformal prediction using a held-out calibration week.

A web dashboard (SKU search bar plus tabs) serves the best-performing model:

- (a) historical and future demand with 95% intervals
- (b) a table for manually overriding the underlying data
- (c) covariate time series

---

## Repository layout

```
demand_imputation_and_forecasting_with_uncertainty/
├── contracts/        FROZEN in week 7. Schemas, interfaces, shared file formats
├── configs/          YAML for data, splits, each model, conformal settings
├── src/uncertainty_aware_planner/
│   ├── data/         download, version record, stockout masks, daily aggregation, splits
│   ├── models/
│   │   ├── imputation/    one file per architecture
│   │   └── forecasting/   one file per architecture
│   ├── uq/           conformal.py
│   ├── evaluation/   metrics, low-sales slice, fairness report, error analysis
│   ├── pipeline/     orchestration, seeds, run log
│   └── flags/        review flag, "demand has changed" warning
├── api/              FastAPI service + planner edit log
├── dashboard/        SKU search and tabs
├── fixtures/         small SYNTHETIC tables that match contracts/CONTRACTS.md
├── experiments/      per-person scratch space (gil/ jeena/ tarakaram/ vishnu/)
├── results/          small committed summaries only (CSV/JSON, figures)
├── slurm/            Explorer cluster job scripts and environment setup
├── tests/            mirrors src/; schema tests run against fixtures
└── docs/             proposal, dev contract, feature cards, storyboard, architecture slides, dataset license
```

The repository is `demand_imputation_and_forecasting_with_uncertainty`; the importable Python package is `uncertainty_aware_planner` (underscores, since hyphens cannot appear in imports).

**Design rule:** streams hand off through files that obey `contracts/`, not through each other's code. Evaluation reads prediction tables only. Conformal prediction wraps any model's quantile output. Each architecture lives in its own file behind a shared interface.

---

## Quick start

The environment is managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock`; Python 3.12 is installed by uv if missing.

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/gbenezer/uncertainty_aware_planner
cd uncertainty_aware_planner
uv sync --all-groups     # .venv with the project, ruff and pytest
uv run pytest            # tests on fixtures; pass with no data downloaded
uv run python -m uncertainty_aware_planner.pipeline.baseline_smoke   # baseline smoke run, CPU, ~10 s
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
git clone https://github.com/gbenezer/uncertainty_aware_planner
cd uncertainty_aware_planner
uv sync --all-groups
uv run pytest
uv run python -m uncertainty_aware_planner.pipeline.baseline_smoke
```

`uv run <cmd>` runs inside `.venv` without activating it. To activate: `source .venv/bin/activate` (macOS / Linux) or `.venv\Scripts\activate` (Windows). Linux and Windows get the CUDA 12.8 build of torch (it also runs on CPU-only machines); macOS gets the standard PyPI build (CPU / Apple MPS).

### Running on the Explorer cluster

Job scripts and environment setup live in `slurm/`. Use a laptop for tests and small samples; use Explorer for full runs. See `slurm/README.md` for module loads and submission examples.

---

## Reproducing the baseline

**Baseline.** The baseline is the non-quantile DLinear imputation model, with the loss and architecture from the dataset paper repository (Feature Cards 1.1). It uses the same split and metric as the planned quantile models:
- **Split:** the six rolling folds in `configs/splits.yaml`.
- **Metric:** daily-pooled WAPE, with WPE for bias.

### Smoke test: one command, laptop CPU, no GPU, no credentials

```bash
uv run python -m uncertainty_aware_planner.pipeline.baseline_smoke
```

It uses the committed SYNTHETIC fixtures, so nothing is downloaded, and takes about 10 seconds. Settings are in `configs/baseline_smoke.yaml`. The run:

1. **Checks the environment, data and split** before training:
   - the Python version against `.python-version`,
   - the fixture SHA-256 hashes against `configs/baseline_smoke.yaml`,
   - `configs/splits.yaml` against the data calendar.

   It stops at the first failure.
2. **Trains the baseline** for 3 seeds × 6 folds, using the same model config, masks and metric as the full-data run.
3. **Compares the result with the log.** Mean WAPE and WPE are checked against the smoke reference rows in `results/imputation/baseline_log.csv`. It prints `MATCH` or `MISMATCH`, and exits with status 1 on a mismatch.

The output ends with:

```
variant     runs  WAPE mean ± sd   WPE mean ± sd     reference WAPE             status
not_leaked    18  0.3076 ± 0.0274  -0.2436 ± 0.0608  0.3076 (commit cfe9d50)    MATCH
leaked        18  0.3202 ± 0.0628  -0.2722 ± 0.0852  0.3202 (commit cfe9d50)    MATCH
```

Run it twice and the numbers are identical: seeds are set before every model build, and torch runs in deterministic mode. Each run writes a run log to `data/runs/baseline_smoke/` with the commit, environment, hashes and per-run results. These are synthetic-data numbers that check the pipeline; they are not the official baseline.

### Official baseline number (full FreshRetailNet-50K)

| Variant | WAPE, mean ± sd | WPE, mean ± sd | Runs | Commit |
|---|---|---|---|---|
| **not_leaked** (official): fit on the fold's train window | **45.69% ± 6.97** | −34.02% ± 12.56 | 9 seeds × 6 folds | `5709f5b` |
| leaked (reference only): fit on the val window's visible hours | 42.07% ± 8.05 | −26.01% ± 17.95 | 9 seeds × 6 folds | `5709f5b` |

Each row of `results/imputation/baseline_log.csv` records the commit, the model config and its SHA-256, the data revision and file SHA-256s, the split config SHA-256, the seeds, the hardware, and the source CSV (`experiments/gil/data/seventh_dlinear_baseline_experiment.csv`).

To regenerate the official number, run this on Explorer (one GPU, about 6 hours) from the repository root:

```bash
uv run python -m uncertainty_aware_planner.data load    # downloads the pinned revision and checks its SHA-256
uv run python -m uncertainty_aware_planner.data masks   # writes data/processed/eval_masks.parquet
mkdir -p experiments/gil/job_logs
# the script skips (seed, fold) pairs already in its results CSV, so move the logged one aside first
mv experiments/gil/data/seventh_dlinear_baseline_experiment.csv experiments/gil/data/seventh_dlinear_baseline_experiment.logged.csv
sbatch experiments/gil/submission_script.sh
```

Reruns of the same seed on Explorer have given identical numbers: seed 86400, fold 1 is 0.51325065 in each of `first_` to `fourth_dlinear_baseline_experiment.csv`.

Other hardware takes a different training path. We re-ran seed 86400, fold 1 on a laptop CPU:

| Variant | Laptop CPU | Logged on GPU |
|---|---|---|
| not_leaked | 42.93% | 42.31% |
| leaked | 49.83% | 43.25% |

Both differences are within the spread across seeds on that fold: 34.6% to 67.4% for not_leaked. Compare full-data numbers on Explorer GPUs only. The smoke test is the exact run-twice check.

---

## Running the pipeline

Each stage reads the previous stage's output and writes to `data/` (gitignored). Settings live in `configs/`; models that train take `--seed`. Run these on a laptop for small checks and on Explorer for full runs (`slurm/README.md`).

```bash
# Data (once per machine), see src/uncertainty_aware_planner/data/README.md
uv run python -m uncertainty_aware_planner.data load      # download + validate the pinned dataset revision
uv run python -m uncertainty_aware_planner.data splits    # fold and final windows
uv run python -m uncertainty_aware_planner.data masks     # in-stock hours hidden for imputation scoring

# Imputation (Feature 1): each owner's PyPOTS model writes data/processed/imputed_hourly/<run-id>.parquet
# (data README Step 3). Until one exists, the "no recovery" baseline stands in for it:
uv run python -m uncertainty_aware_planner.pipeline.censored    # writes imputed_hourly/imp-none-censored.parquet

# Daily demand from one imputation run
uv run python -m uncertainty_aware_planner.data daily \
    --imputed data/processed/imputed_hourly/<run-id>.parquet --run-id <run-id>

# Forecasting (Feature 2): one seed and one fold per run; omit --fold for the final model, add --smoke for a tiny CPU check
uv run python -m uncertainty_aware_planner.pipeline.forecast \
    --config configs/forecasting_models/temporal_fusion_transformer.yaml \
    --demand-daily data/processed/demand_daily/<run-id>.parquet --seed 2025 --fold 1

# Dashboard (reads demand_daily)
uv run streamlit run dashboard/app.py
```

Conformal calibration, evaluation and end-of-project data cleanup do not have a command yet; they are in progress.

Every run writes a run-log record (data version, git commit, config, seed) so any result can be reproduced.

---

## Data

**Primary dataset:** [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K) (Dingdong Ltd.), licensed CC BY 4.0.

- 50,000 store-product series, 898 stores, 18 cities, about 863 fresh products
- Roughly 90 training days and 7 evaluation days per series, March to June 2024
- Hourly sales and hourly out-of-stock flags, plus discount, holiday, activity, and daily weather
- Stores, cities, and products appear only as anonymized codes
- Sales are globally normalized by an undisclosed coefficient, so forecasts are not in real units

**Fields we keep:** codes for city, store, product, and product group; date; daily and hourly sales; hourly stockout flags; discount rate; holiday and activity flags; daily weather. We add no outside data, and we do not try to identify what any code stands for.

**Known gaps:** no prices or costs, no stock on hand, no lead times, no product names, no launch dates. Low-selling products have many zeros and more error.

**Data version.** Raw data is never committed. `configs/data.yaml` records:
- the source,
- the pinned Hub revision,
- the license,
- the first download date,
- the SHA-256 and size of both raw files.

`load` stops if a download does not match those hashes. Each machine also writes its own record to `data/raw/DATA_VERSION.json`, which is gitignored.

If you delete `data/`, run `uv run python -m uncertainty_aware_planner.data load` again to get the same data back. It re-downloads the pinned revision and checks the hashes; the dataset is public, so no credentials are needed. License terms, attribution and our changes to the data are in `docs/DATASET_LICENSE.md`.

---

## Experiment design

All architectures are compared under identical conditions. These settings live in `configs/splits.yaml`, not in code.

- **Splits:** rolling-origin by time (train on 4 weeks, evaluate on the next 4 weeks, 6 random cut-offs). The last training week is the conformal calibration set. Random splits are not used, to avoid leakage.
- **Imputation scoring:** non-stockout hours only, since no ground truth exists for censored hours. Stockout (zero-valued) data is excluded from training and metrics.
- **Seeds:** at least 3.
- **Final evaluation:** train on the majority of the official training split and evaluate on the canonical test split (week 15).
- **Hyperparameters:** kept as close to the dataset paper as possible.

### Metrics and targets

| Metric | Role | Target |
|---|---|---|
| WAPE | Error magnitude | Within 1 SD of published baseline (imputation 29.99%, forecasting 30.60%) |
| WPE | Bias direction | Comparable to baseline (imputation -1.28%, forecasting +4.02%) |
| PICP | Calibration | Mean above 80%; 95% aim by construction and conformalization |
| MPIW | Sharpness | Lowest among qualifying models |
| MASE | Client-requested | Informational and tie-breaker |

**Decision rule:** the model that meets the WAPE and PICP requirements with the lowest MPIW, with WPE and MASE as tie-breakers. If nothing qualifies, conformalized DLinear is the reference and is deployed with its performance stated plainly.

**Slice:** the low-sales slice is store-product pairs with mean scaled demand below 1, defined as in the dataset authors' repository.

### Leakage checks

The table lists where test information could reach training, and what the code does about each case.

| Risk | What the code does |
|---|---|
| A fold's val window inside its own training data | **Split by date.** Train is the 28 days before the cut-off; val is the 28 days from it (`data/splits.py`, `fold_windows`). `validate_splits` stops if splits overlap, leave the data range, or let a fold touch the calibration or test week. |
| Calibration or test week in imputation training | **Held out.** The last fold ends 2024-06-09. The final imputation model trains on 03-28..06-18, before the calibration week 06-19..06-25 and the test week 06-26..07-02. |
| Fitting on the val window's visible hours (the `leaked` variant) | **Kept apart.** It is logged separately and never used as the official number. The model then sees the evaluation period's sales pattern, though not the hidden hours, and WAPE drops by 3.6 points (42.07% vs 45.69%). |
| True values of hidden hours | **Never passed to `fit`.** Only in-stock hours on fully observed days are hidden, with a fixed seed (`data/masks.py`). Their true values stay in `X_ori`, which PyPOTS `fit` and `predict` do not read. Scoring counts only open-hour days that have hidden hours and no real stockout (`evaluation/error_metrics.py`). |
| Preprocessing fitted across splits | **Nothing is fitted.** Discount cleaning uses fixed constants. Covariates are scaled within each window by that window's own maximum. Sales are not scaled. |
| Duplicate rows | **Rejected.** `validate_calendar` stops on duplicate (store, product, date) rows. |
| Features that encode the label | **Not used.** Stockout flags are not model inputs; stockout hours enter only as missing sales. |
| Overlapping folds | **Open caveat, not leakage.** The six cut-offs fall within 14 days, so folds share days. The ± across folds therefore understates the true uncertainty. |
| Forecasting: observed weather as a "known future" input | **Open risk.** `configs/data.yaml` gives the models the actual weather of the 7 forecast days, as the dataset authors do. A deployed model would only have a weather forecast, so forecast errors are optimistic. |

---

## Ownership

| Area (under `src/uncertainty_aware_planner/` unless noted) | Lead |
|---|---|
| `data/`, `uq/conformal.py`, DLinear imputation (quantile) | Tarakaram |
| `evaluation/`, fairness report, TimesNet imputation, DLinear baselines | Gil |
| `api/`, `dashboard/` (both at repo root), DLinear forecasting (quantile) | Jeena |
| `pipeline/`, `slurm/` (repo root), TFT forecasting, final runs, security check | Vishnu |
| `contracts/` (repo root) | All four; changes need every lead's approval |

Ownership is agreed in this table. There is no `CODEOWNERS` file yet, so reviewers are requested by hand. Each lead also writes one section of the final report.

---

## Working agreements

- **Contracts are frozen in week 7.** Shared tables are defined in `contracts/CONTRACTS.md`. Changing one is a PR that all four leads approve.
- **Code to the interface, test on fixtures.** Use `fixtures/` until real outputs exist. Do not import another person's model code.
- **Branches:** `main` is protected. Open PRs from `feat/<name>-<topic>`; one review from a different lead and passing CI are required.
- **Configs define experiments.** Do not hard-code cut-offs, seeds, or quantile levels.
- **Conformal prediction is the priority UQ method.** Test-time augmentation and the external interval network are stretch goals and must not block anything.
- **Planner edits never touch the dataset.** Edits go to a separate edit log (store, product, day, old value, new value, date, reason). Edits take effect at the next forecast run; the model is not retrained after each edit.
- **Report by group.** No result is final until it is shown separately for low- and high-selling products.
- **Commit only small things.** `results/` and `fixtures/` only. No raw data, checkpoints, or parquet files elsewhere.
- **Scope changes** go through the written amendment process in the development contract, Section 5, not through a code change.


### Acceptance criteria (checked at the showcase)

1. Four models (two imputation, two forecasting) are evaluated on the agreed metrics.
2. All models perform within 1 SD of published WAPE across 3 or more folds.
3. All models have mean PICP above 80%.
4. Deployed models have the lowest MPIW.
5. The web interface has an SKU search bar and the three tabs described above.

---

## Out of scope

- Live ordering, ERP, or store-system integration; streaming data
- Delivery-time or lead-time forecasting
- Stock decisions: no safety stock, reorder points, minimum stock, order amounts, or stock-level charts
- Explanation scores (such as SHAP), "main reasons", per-product past coverage, or low-data flags in the dashboard
- Hourly charts (models use hourly data; the dashboard shows daily views)
- Live weather, private company data, pricing decisions, ranking stores or staff, or de-anonymizing codes

---

## Attribution

Data: FreshRetailNet-50K by Dingdong Ltd. and collaborators, CC BY 4.0. License terms and the changes we make are in `docs/DATASET_LICENSE.md`.

Baselines and model implementations follow the dataset paper (Wang et al., arXiv:2505.16319) and its [baseline repository](https://github.com/Dingdong-Inc/frn-50k-baseline).

AI usage is disclosed in the project proposal (Appendix F).
