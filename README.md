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
4. [Running the pipeline](#running-the-pipeline)
5. [Data](#data)
6. [Experiment design](#experiment-design)
7. [Ownership](#ownership)
8. [Working agreements](#working-agreements)
9. [Out of scope](#out-of-scope)
10. [Attribution](#attribution)

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
├── src/uncertainty-aware-planner/
│   ├── data/         download, version record, stockout masks, daily aggregation, splits
│   ├── models/
│   │   ├── imputation/    one file per architecture
│   │   └── forecasting/   one file per architecture
│   ├── uq/           conformal.py
│   ├── evaluation/   metrics, low-sales slice, fairness report, error analysis
│   ├── pipeline/     orchestration, seeds, run log
│   └── flags/        confidence labels, "demand has changed" warning
├── api/              FastAPI service + planner edit log
├── dashboard/        SKU search and tabs
├── fixtures/         small SYNTHETIC tables that match contracts/schemas.py
├── experiments/      per-person scratch space (gil/ jeena/ tarakaram/ vishnu/)
├── results/          small committed summaries only (CSV/JSON, figures)
├── slurm/            Explorer cluster job scripts and environment setup
├── tests/            mirrors src/; schema tests run against fixtures
└── docs/             experiment cards, datasheet, governance plan, architecture figure
```

**Design rule:** streams hand off through files that obey `contracts/`, not through each other's code. Evaluation reads prediction tables only. Conformal prediction wraps any model's quantile output. Each architecture lives in its own file behind a shared interface.

---

## Quick start

```bash
git clone <repo-url> uncertainty-aware-planner
cd uncertainty-aware-planner

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make test        # schema tests on fixtures; should pass with no data downloaded
```

### Running on the Explorer cluster

Job scripts and environment setup live in `slurm/`. Use a laptop for tests and small samples; use Explorer for full runs. See `slurm/README.md` for module loads and submission examples.

---

## Running the pipeline

Each stage reads the previous stage's output and writes to `data/` (gitignored). Every stage accepts `CONFIG=<path>` and `SEED=<int>`.

```bash
make data        # download FreshRetailNet-50K, record version/date, build series, mask stockouts, make splits
make impute      # Feature 1: train and run imputation models (quantile outputs)
make forecast    # Feature 2: train and run forecasting models on observed + imputed demand
make conformal   # calibrate intervals on the held-out calibration week
make eval        # metrics, low-sales slice, fairness report, error analysis
make serve       # start API and dashboard on the selected model
make clean-data  # delete raw data, checkpoints, and intermediates (see end-of-project handling)
```

Quick check on fixtures, without downloading anything:

```bash
make serve FIXTURES=1
```

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

Raw data is never committed. `data.yaml` records the download date and dataset version.

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

---

## Ownership

| Area | Lead |
|---|---|
| `src/frn/data/`, fairness report, DLinear imputation (quantile) | Tarakaram |
| `src/frn/evaluation/`, `src/frn/uq/conformal.py`, TimesNet imputation, DLinear baselines | Gil |
| `api/`, `dashboard/`, DLinear forecasting (quantile) | Jeena |
| `src/frn/pipeline/`, `slurm/`, TFT forecasting, final runs, security check | Vishnu |
| `contracts/` | All four; changes need every lead's approval |

Ownership is enforced through `CODEOWNERS`. Each lead also writes one section of the final report.

---

## Working agreements

- **Contracts are frozen in week 7.** Shared tables are defined in `contracts/schemas.py` and described in `contracts/CONTRACTS.md`. Changing one is a PR that all four leads approve.
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

Data: FreshRetailNet-50K by Dingdong Ltd. and collaborators, CC BY 4.0. Any changes we made are described in `docs/datasheet.md`.

Baselines and model implementations follow the dataset paper (Wang et al., arXiv:2505.16319) and its [baseline repository](https://github.com/Dingdong-Inc/frn-50k-baseline).

AI usage is disclosed in the project proposal (Appendix F).
