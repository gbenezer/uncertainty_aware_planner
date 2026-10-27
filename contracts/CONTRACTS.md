# CONTRACTS.md

Shared data formats and interfaces for the Uncertainty-Aware Demand and Inventory Planning Tool.

**Status:** DRAFT. To be reviewed by all four leads and **frozen at the Week 7 Midpoint Design Review (Oct 27, 2026)**.
**After freeze:** any change is a PR that every lead approves, with `CONTRACT_VERSION` bumped in `contracts/schemas.py`.

This file is the human-readable description. `contracts/schemas.py` (pandera/pydantic) is the enforced version; if they disagree, fix the disagreement in the same PR. `contracts/interfaces.py` holds the abstract classes.

---

## 1. Conventions

| Item | Rule |
|---|---|
| File format | Parquet for tables, YAML for configs, JSON for run logs |
| Keys | `store_id`, `product_id` are the dataset's anonymized codes, stored as strings. Never decoded or joined to outside data |
| Time | `date` is a calendar date (no timezone). `hour` is an integer 0 to 23. Hourly tables carry both |
| Units | Demand is in the dataset's **normalized** sales units. Never describe it as real units or currency |
| Quantile columns | Named `q<level>` with level as 3-digit per-mille, e.g. `q025`, `q500`, `q975`. Always monotone non-decreasing across levels (enforce after prediction) |
| Missing values | Null, never 0, for "not computed". A 0 in a demand column means zero demand |
| Splits | `split` is one of `train`, `val`, `calib`, `test`. Rolling-origin folds also carry `fold` (int) |
| Seeds | `seed` is an int on every prediction table |
| Identifiers | `model_id` is `<feature>-<architecture>-<variant>`, e.g. `imp-timesnet-quantile`, `fc-tft-quantile` |
| Paths | Under `data/` (gitignored). Layout in section 6 |
| Sort order | Tables are sorted by their primary key before writing |

### Quantile levels

| Interval | Lower | Upper | Source |
|---|---|---|---|
| 95% | `q025` | `q975` | Models trained with pinball loss at these levels (experiment cards, meeting notes) |

---

## 2. Tables

### 2.1 `series_hourly` (Data lead to Imputation models)

One row per store-product-hour.

**Primary key:** `store_id, product_id, date, hour`

| Column | Type | Notes |
|---|---|---|
| `store_id`, `product_id` | str | Anonymized codes |
| `city_id`, `group_id` | str | Management group and category levels carried as `cat1_id`, `cat2_id`, `cat3_id` if used |
| `date`, `hour` | date, int8 | |
| `sales` | float | Normalized hourly sales. Observed |
| `is_stockout` | bool | True when the hourly out-of-stock flag is 1 |
| `discount`, `holiday_flag`, `activity_flag` | float, int8, int8 | Daily values repeated across hours |
| `precipitation`, `temp_avg`, `humidity`, `wind_level` | float | **Daily** weather repeated across hours (not hourly) |
| `split`, `fold` | str, int8 | See conventions |

**Invariants:** no duplicate keys. `is_stockout` rows have `sales` treated as censored, not as demand. Rows with `is_stockout = True` are excluded from imputation training and metric calculation.

### 2.2 `imputed_hourly` (Imputation models to Data lead / Forecast models)

One row per store-product-hour **that was a stockout hour** (observed hours are not duplicated here).

**Primary key:** `store_id, product_id, date, hour, model_id, seed, fold`

| Column | Type | Notes |
|---|---|---|
| `store_id`, `product_id`, `date`, `hour` | | As above |
| `model_id`, `seed`, `fold`, `split` | | |
| `q025`, `q975` | float | Raw model quantile outputs |
| `point` | float | Point estimate (see O1) |
| `q025_cal`, `q975_cal` | float, nullable | After conformal calibration. Null before `make conformal` |

**Invariants:** `q025 <= point <= q975`. All values `>= 0` (clip, and record clipping rate in the run log).

### 2.3 `demand_daily` (Data lead to Forecast models)

One row per store-product-day: observed plus recovered demand. This is the forecasting input and is built from `series_hourly` and one chosen `imputed_hourly` run.

**Primary key:** `store_id, product_id, date, imputation_run_id`

| Column | Type | Notes |
|---|---|---|
| `store_id`, `product_id`, `date` | | |
| `imputation_run_id` | str | Which imputation run filled the stockouts. **Forecasts must be reproducible from this**, so versioned copies are kept (experiment card, technical debt) |
| `demand_observed` | float | Sum over non-stockout hours |
| `demand_recovered` | float | Sum of `point` over stockout hours. 0 if no stockout hours |
| `demand_total` | float | `demand_observed + demand_recovered`. The forecast target and input |
| `n_stockout_hours` | int8 | Count of stockout hours that day (0 to 24) |
| `recovered_lo`, `recovered_hi` | float, nullable | Summed interval bounds over stockout hours (see O3) |
| covariates | | `discount`, `holiday_flag`, `activity_flag`, `precipitation`, `temp_avg`, `humidity`, `wind_level` |
| `split`, `fold` | | |

### 2.4 `forecast` (Forecast models to UQ / Evaluation / API)

One row per store-product-forecast-day.

**Primary key:** `store_id, product_id, origin_date, horizon, model_id, seed, fold`

| Column | Type | Notes |
|---|---|---|
| `store_id`, `product_id` | | |
| `origin_date` | date | Last day of history used |
| `horizon` | uint8 | 1 to 7 |
| `target_date` | date | `origin_date + horizon` days |
| `model_id`, `seed`, `fold`, `split` | | |
| `imputation_run_id` | str | Provenance |
| `q025`, `q975` | float | Raw quantile outputs |
| `point` | float | Point forecast (O1) |
| `actual` | float, nullable | `demand_total` for `target_date`, when it exists. **Only the evaluation code reads this column** |

**Invariants:** `horizon in 1..7`. Quantiles monotone. Every `(store, product, origin_date)` has all 7 horizons.

### 2.5 `forecast_calibrated` (UQ to Evaluation / API)

Same key and columns as `forecast`, plus:

| Column | Type | Notes |
|---|---|---|
| `lo95`, `hi95` | float | Conformalized 95% interval |
| `calib_window_start`, `calib_window_end` | date | The held-out calibration week used |
| `qhat95`| float | Conformal adjustment applied |

**Invariants:** `lo95 <= point <= hi95` (if violated, set to the nearest bound and count it in the run log). Intervals must not be calibrated on any data that appears in the scoring window.

### 2.6 `review_output` (API-facing shared table)

**This is the single table the API reads.** One row per `(store_id, product_id, date)`, covering history and the next 7 days of the selected model. The API never reads model outputs directly.

**Primary key:** `store_id, product_id, date`

| Column | Type | Notes |
|---|---|---|
| `store_id`, `product_id`, `date` | | |
| `kind` | str | `observed`, `recovered`, or `forecast` |
| `point` | float | Observed or recovered demand, or forecast |
| `lo95`, `hi95` | float, nullable | Null for plain observed days with no stockout |
| `n_stockout_hours` | int8 | |
| `change_warning` | bool | "Demand has changed" |
| `review_flag` | bool | Range unusually wide, or recent demand repeatedly outside range |
| `model_id`, `run_id` | str | Provenance |
| covariates | | As in `demand_daily`, for the covariate tab |

**Display rules (from the contract):** estimates in `recovered` rows are always shown with their range and never labeled as real sales. The dashboard shows daily views only; no hourly charts, explanation scores, or per-product past-coverage panels.

### 2.7 `edit_log` (API-owned, append-only)

Planner overrides. **The original dataset is never modified.** Edits are applied as an overlay at the next forecast run.

**Primary key:** `edit_id`

| Column | Type | Notes |
|---|---|---|
| `edit_id` | str | UUID |
| `store_id`, `product_id`, `date` | | |
| `field` | str | `demand` or a covariate name |
| `old_value`, `new_value` | float | |
| `edited_at` | timestamp (UTC) | |
| `reason` | str | **Required, non-empty** |
| `applied_in_run_id` | str, nullable | Filled once a forecast run consumes the edit |

**Rules:** append-only (corrections are new rows). The pipeline reads the log, applies the overlay to produce `demand_daily`, and records the set of `edit_id`s used in the run log.

### 2.8 `metrics` (Evaluation output)

Long format so every architecture and slice is comparable.

**Primary key:** `feature, model_id, fold, seed, slice, metric`

| Column | Type | Notes |
|---|---|---|
| `feature` | str | `imputation` or `forecasting` |
| `model_id`, `fold`, `seed`, `split` | | |
| `slice` | str | `all`, `low_sales`, `high_sales` |
| `metric` | str | `wape`, `wpe`, `mase`, `picp95`, `mpiw95`|
| `value` | float | |
| `n` | int | Number of points scored |
| `calibrated` | bool | Whether intervals were conformalized |

**Slice definition:** `low_sales` is store-product pairs with mean scaled demand below 1, defined as in the dataset authors' repository. Every other pair is `high_sales`. Slice membership is computed once and stored in `slices.parquet` so every consumer uses the same split.

---

## 3. Interfaces (`contracts/interfaces.py`)

Each architecture lives in its own file and implements one of these. Constructors take only a config dict; no architecture reads another's code.

```python
class Imputer(ABC):
    def fit(self, train: pd.DataFrame, config: dict, seed: int) -> None: ...
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return imputed_hourly rows (2.2) for stockout hours in df."""
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Imputer": ...

class Forecaster(ABC):
    def fit(self, train: pd.DataFrame, config: dict, seed: int) -> None: ...
    def predict(self, history: pd.DataFrame, origin_dates) -> pd.DataFrame:
        """Return forecast rows (2.4): all 7 horizons per origin."""
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Forecaster": ...

class IntervalWrapper(ABC):
    def calibrate(self, calib_preds: pd.DataFrame, config: dict) -> None:
        """Fit on the held-out calibration week only."""
    def apply(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Return forecast_calibrated rows (2.5) or imputed rows with *_cal filled."""
```

**Rules**

- A `Forecaster` or `Imputer` returns **raw quantiles only**. Calibration happens in an `IntervalWrapper`, so conformal prediction, test-time augmentation, and the external interval network are interchangeable.
- `predict` is deterministic given a seed and loaded weights.
- Evaluation takes prediction tables and the actuals; it must never import from `models/` or `uq/`.

---

## 4. Config contract

| File | Required keys |
|---|---|
| `configs/data.yaml` | `dataset_name`, `dataset_version`, `download_date`, `fields_kept` |
| `configs/splits.yaml` | `calib_weeks`, `train_weeks`, `eval_weeks`, `n_cutoffs`, `cutoff_seed`, `seeds` |
| `configs/imputation/*.yaml`, `configs/forecasting/*.yaml` | `model_id`, `architecture`, `quantiles`, `hyperparameters`, `covariates`, `lag_features` |
| `configs/conformal.yaml` | `alphas`, `score_function`, `calib_source` |

**Controlled conditions:** all architectures compared for one feature must read the **same** `splits.yaml`, `covariates`, and `lag_features`. The pipeline refuses to run a comparison if they differ.

---

## 5. Run log (`runs/<run_id>/run_log.json`)

Written by `pipeline/run_log.py` for every stage of every run. One record per stage.

```json
{
  "run_id": "2026-11-03T14-22-10_imp-timesnet-quantile_s1",
  "stage": "impute",
  "model_id": "imp-timesnet-quantile",
  "seed": 1,
  "fold": 2,
  "dataset_version": "<from data.yaml>",
  "download_date": "<from data.yaml>",
  "git_commit": "<hash>",
  "git_dirty": false,
  "config_hash": "<sha256 of resolved configs>",
  "contract_version": "1.0",
  "inputs": ["<paths or run_ids>"],
  "outputs": ["<paths>"],
  "edit_ids_applied": [],
  "clipped_fraction": 0.0,
  "host": "laptop|explorer",
  "started_at": "...", "finished_at": "..."
}
```

A run with `git_dirty: true` is allowed for experiments but is **not eligible** for final results.

---

## 6. Paths

```
data/
  raw/                         downloaded dataset (never committed)
  interim/series_hourly.parquet
  interim/slices.parquet
  runs/<run_id>/
    imputed_hourly.parquet
    demand_daily.parquet
    forecast.parquet
    forecast_calibrated.parquet
    run_log.json
  serving/review_output.parquet      selected model only
  serving/edit_log.parquet           API-owned
results/
  metrics.parquet  (or csv for committed summaries)
  fairness/        by-group tables and figures
fixtures/                      synthetic tables for each of 2.1 to 2.8
```

---

## 7. Who produces and consumes what

| Table | Produced by | Consumed by |
|---|---|---|
| `series_hourly`, `slices` | Data (Tarakaram) | Imputation models, Evaluation |
| `imputed_hourly` | Imputation models (Tarakaram, Gil) | Data, UQ, Evaluation |
| `demand_daily` | Data (Tarakaram), reading `edit_log` | Forecast models |
| `forecast` | Forecast models (Jeena, Vishnu) | UQ, Evaluation |
| `forecast_calibrated` | UQ (Tarakaram) | Evaluation, flags, API |
| `review_output` | Pipeline (Vishnu) with `flags/` | API and dashboard (Jeena) |
| `edit_log` | API (Jeena) | Data (overlay) |
| `metrics` | Evaluation (Gil) | Pipeline (model selection), reports |

**Model selection** (Vishnu's pipeline) reads only `metrics` and applies the decision rule: qualifying WAPE and PICP, then lowest MPIW, with WPE and MASE as tie-breakers.

---

## 8. Validation

- `tests/test_schemas.py` validates every fixture against `schemas.py`.
- The pipeline validates every table on write and on read. A validation failure stops the stage.
- CI fails on any change to `contracts/` that does not bump `CONTRACT_VERSION`.
- Fixtures are **synthetic**. Never commit real dataset rows as fixtures.

---

## 9. Open decisions (resolve before the Week 7 freeze)

| # | Question | Context | Proposed default |
|---|---|---|---|
| **O1** | What is `point`? | Meeting notes derive the prediction from the interval midpoint; a median (`q500`) is the more standard choice | Midpoint of the raw 95% bounds, to match the meeting notes, with `q500` added later only if a model trains it. Decide once and apply to every architecture |
| **O2** | Baseline model output | The baseline is not a quantile model, so it has no interval | Baseline rows have null `q*` and `lo/hi`, and are scored on WAPE, WPE, and MASE only |
| **O3** | Fold count | Proposal and cards say 6 random cut-offs; contract says 3-fold or more | Use 6 cut-offs in `splits.yaml`, which satisfies "3 or more" |
| **O4** | Clipping at zero | Quantile heads can go negative | Clip at 0 after prediction and log the fraction |

---

## 10. Change log

| Version | Date | Change | Approved by |
|---|---|---|---|
| 0.1 | 2026-10-27 | Initial draft | pending |
