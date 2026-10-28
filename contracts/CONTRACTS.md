# CONTRACTS.md

Shared data formats and interfaces for the Uncertainty-Aware Demand and Inventory Planning Tool.

**Status:** DRAFT. To be reviewed by all four leads and **frozen at Week 7**.
**After freeze:** any change is a PR that every lead approves, with `CONTRACT_VERSION` bumped in `contracts/schemas.py`.

This file is the human-readable description. `contracts/schemas.py` (pandera/pydantic) is the enforced version; if they disagree, fix the disagreement in the same PR. `contracts/interfaces.py` holds the abstract classes.

---

## 1. Conventions

| Item | Rule |
| --- | --- |
| File format | Parquet for tables, YAML for configs, JSON for run logs |
| Keys | `store_id`, `product_id` are the dataset's anonymized integer codes, stored as int64. Never decoded or joined to outside data |
| Time | `date` is a calendar date (no timezone) in year-month-day format. Hourly tables carry `date`, `time` (naive time of day) and `datetime` (naive datetime). All are timezone-independent |
| Units | Demand is in the dataset's **normalized** sales units. Never describe it as real units or currency |
| Quantile columns | Named `q<level>` with level as 3-digit per-mille, e.g. `q025`, `q500`, `q975`. Always monotone non-decreasing across levels (enforce after prediction) |
| Missing values | Null, never 0, for "not computed"|
| Splits | `split` is one of `train`, `val`, `calib`, `test`. Rolling-origin folds also carry `fold` (int) |
| Seeds | `seed` is an int on every prediction table |
| Identifiers | `model_id` is `<feature>-<architecture>-<variant>`, e.g. `imp-timesnet-quantile`, `fc-tft-quantile` |
| Run IDs | `run_id` is `<model_id>-<seed>`|
| Paths | Under `data/` (gitignored). Layout in section 6 |

### Quantile levels

|Interval|Lower|Upper|Source|
|---|---|---|---|
|95%|`q025`|`q975`|Models trained with pinball loss at these levels (experiment cards, meeting notes)|

---

## 2. Tables

### 2.1 `raw_series`

One row per store-product-day, with sequences. This is the format of the raw data without modification.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`dt`|string|The date|
|`sale_amount`|float64|The daily sales quantity after global normalization (Multiplied by a specific coefficient)|
|`hours_sale`|Sequence(float64)\[24\]|The hourly sales quantity after global normalization (Multiplied by a specific coefficient)|
|`stock_hour6_22_cnt`|int32|The number of out-of-stock hours between 6:00 and 22:00 (store hours)|
|`hours_stock_status`|Sequence(int32)\[24\]|The hourly out-of-stock status|
|`discount`|float64|The discount rate (1.0 means no discount, 0.9 means 10% off)|
|`holiday_flag`|int32|Holiday indicator|
|`activity_flag`|int32|Activity indicator|
|`precpt`|float64|The total precipitation|
|`avg_temperature`|float64|The average temperature|
|`avg_humidity`|float64|The average humidity|
|`avg_wind_level`|float64|The average wind force|

#### Hierarchical structure

1. warehouse: city_id > store_id
2. product category: management_group_id > first_category_id > second_category_id > third_category_id > product_id

### 2.2 `series_hourly`

One row per store-product-hour, after sequence disaggregation, removal of daily sales summary, and conversion to datetime information. `is_stockout` rows have `scaled_sales_quantity` treated as censored, not as demand. Rows with `is_stockout = True` should be excluded from imputation training and metric calculation.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`date`|datetime.date|The date|
|`time`|datetime.time|The naive (timezone-independent) time|
|`datetime`|datetime.datetime|The naive (timezone-independent) datetime|
|`scaled_sales_quantity`|float64|The hourly sales quantity after global normalization (Multiplied by a specific coefficient)|
|`is_stockout`|bool|Out-of-stock hour indicator variable|
|`discount`|float64|The discount rate (1.0 means no discount, 0.9 means 10% off)|
|`holiday_flag`|int32|Holiday indicator|
|`activity_flag`|int32|Activity indicator|
|`precpt`|float64|The total precipitation|
|`avg_temperature`|float64|The average temperature|
|`avg_humidity`|float64|The average humidity|
|`avg_wind_level`|float64|The average wind force|


### 2.3 `imputed_hourly`

One row per store-product-hour **that was a stockout hour**. Observed hours are not duplicated here, with one exception: in calibration and evaluation runs, in-stock hours that were deliberately masked and imputed are also included, carrying their true sales in `actual_demand`.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`date`|datetime.date|The date|
|`time`|datetime.time|The naive (timezone-independent) time|
|`datetime`|datetime.datetime|The naive (timezone-independent) datetime|
|`impute_model_id`|string|Imputation model identifier as defined in Conventions|
|`impute_seed`|uint|Random seed used in model run|
|`impute_fold`|uint, nullable|Fold membership for rolling fold splits, null for calibration and test|
|`impute_split`|string or enum|One of `train`, `val`, `calib`, `test`|
|`q025`|float, nullable|Raw model lower quantile outputs. Null only for non-quantile baseline models|
|`q975`|float, nullable|Raw model upper quantile outputs. Null only for non-quantile baseline models|
|`estimated_demand`|float|Estimated/imputed demand (midpoint of quantiles)|
|`actual_demand`|float, nullable|True sales for a masked in-stock hour. **Null for real stockout hours**, where no ground truth exists. Read only by evaluation and calibration code|
|`q025_cal`|float, nullable|Lower quantile output after conformal calibration. Null before conformalization|
|`q975_cal`|float, nullable|Upper quantile output after conformal calibration. Null before conformalization|
|`qhat95`|float, nullable|Conformal adjustment applied|

**Invariants:** `q025 <= estimated_demand <= q975`. All estimated values `>= 0` (clip, and record clipping rate in the run log). Rows with non-null `actual_demand` are masked evaluation rows and never feed `demand_daily`.

### 2.4 `demand_daily`

One row per store-product-day: observed plus recovered demand. This is the forecasting input and is built from `series_hourly` and one chosen `imputed_hourly` run.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`date`|datetime.date|The date|
|`imputation_run_id`|str|Which imputation run filled the stockouts. **Forecasts must be reproducible from this**, so versioned copies of `imputed_hourly` are kept|
|`demand_observed`|float|Sum over non-stockout hours|
|`demand_recovered`|float|Sum of `estimated_demand` over stockout hours (rows with null `actual_demand`). 0 if no stockout hours|
|`demand_total`|float|`demand_observed + demand_recovered`. The forecast target and input|
|`stock_hour6_22_cnt`|int32|The number of out-of-stock hours between 6:00 and 22:00 (store hours)|
|`discount`|float64|The discount rate (1.0 means no discount, 0.9 means 10% off)|
|`holiday_flag`|int32|Holiday indicator|
|`activity_flag`|int32|Activity indicator|
|`precpt`|float64|The total precipitation|
|`avg_temperature`|float64|The average temperature|
|`avg_humidity`|float64|The average humidity|
|`avg_wind_level`|float64|The average wind force|
|`forecast_fold`|uint, nullable|Fold membership for rolling fold splits, null for calibration and test|
|`forecast_split`|string or enum|One of `train`, `val`, `calib`, `test`|

### 2.5 `forecast`

One row per store-product-forecast-day.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`date`|datetime.date|The date|
|`imputation_run_id`|str|Which imputation run filled the stockouts.|
|`forecast_model_id`|string|Forecast model identifier as defined in Conventions|
|`forecast_seed`|uint|Random seed used in model run|
|`forecast_fold`|uint, nullable|Fold membership for rolling fold splits, null for calibration and test|
|`forecast_split`|string or enum|One of `train`, `val`, `calib`, `test`|
|`forecast_run_id`|str|Which forecast run generated the data.|
|`origin_date`|date|Last day of history used|
|`horizon`|uint8|1 to 7|
|`target_date`|date|`origin_date + horizon` days|
|`q025`|float, nullable|Raw model lower quantile outputs. Null only for non-quantile baseline models|
|`q975`|float, nullable|Raw model upper quantile outputs. Null only for non-quantile baseline models|
|`estimated_demand`|float|Forecasted daily demand (midpoint of quantiles)|
|`actual_demand`|float, nullable|`demand_total` for `target_date`, when it exists (is in the past). **Only the evaluation code reads this column**|
|`q025_cal`|float, nullable|Lower quantile output after conformal calibration. Null before conformalization|
|`q975_cal`|float, nullable|Upper quantile output after conformal calibration. Null before conformalization|
|`qhat95`|float, nullable|Conformal adjustment applied|


### 2.6 `imputation_output` (API-facing table)

The table read for the historical hourly imputation time series tab; one row per store-product-hour.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`date`|datetime.date|The date|
|`time`|datetime.time|The naive (timezone-independent) time|
|`datetime`|datetime.datetime|The naive (timezone-independent) datetime|
|`kind`|str or enum|`observed` or `recovered`|
|`demand`|float|Estimated or actual demand|
|`q025_cal`|float, nullable|Lower quantile output after conformal calibration. Null for observed data|
|`q975_cal`|float, nullable|Upper quantile output after conformal calibration. Null for observed data|

### 2.7 `forecasting_output` (API-facing table)

The table read for the 7-day daily forecast time series tab; one row per store-product-forecast-day.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`origin_date`|date|Last day of history used|
|`horizon`|uint8|1 to 7|
|`target_date`|date|`origin_date + horizon` days|
|`date`|datetime.date|The date|
|`demand`|float|Forecasted demand or actual demand (for origin date)|
|`q025_cal`|float, nullable|Lower quantile output after conformal calibration.|
|`q975_cal`|float, nullable|Upper quantile output after conformal calibration.|

### 2.8 `covariate_output` (API-facing table)

The table read for the covariate tab; one row per historical store-product-day.

|Column|Type|Notes|
|---|---|---|
|`city_id`|int64|The encoded city ID|
|`store_id`|int64|The encoded store ID|
|`management_group_id`|int64|The encoded management group ID|
|`first_category_id`|int64|The encoded first category ID|
|`second_category_id`|int64|The encoded second category ID|
|`third_category_id`|int64|The encoded third category ID|
|`product_id`|int64|The encoded product ID|
|`date`|datetime.date|The date|
|`discount`|float64|The discount rate (1.0 means no discount, 0.9 means 10% off)|
|`holiday_flag`|int32|Holiday indicator|
|`activity_flag`|int32|Activity indicator|
|`precpt`|float64|The total precipitation|
|`avg_temperature`|float64|The average temperature|
|`avg_humidity`|float64|The average humidity|
|`avg_wind_level`|float64|The average wind force|

### 2.9 `edit_log` (API-owned, append-only)

Planner overrides. **The original dataset is never modified.** Edits are applied as an overlay at the next forecast run.

**Primary key:** `edit_id`

| Column | Type | Notes |
| --- | --- | --- |
| `edit_id` | str | UUID |
| `store_id`, `product_id`, `date` | | |
| `field` | str | `demand` or a covariate name |
| `old_value`, `new_value` | float | |
| `edited_at` | timestamp (UTC) | |
| `reason` | str | **Required, non-empty** |
| `applied_in_run_id` | str, nullable | Filled once a forecast run consumes the edit |

**Rules:** append-only (corrections are new rows). The pipeline reads the log, applies the overlay to produce `demand_daily`, and records the set of `edit_id`s used in the run log.

### 2.10 `metrics` (Evaluation output)

Long format so every architecture and slice is comparable.

**Primary key:** `feature, model_id, fold, seed, split, slice, metric, calibrated`

| Column | Type | Notes |
| --- | --- | --- |
| `feature` | str | `imputation` or `forecasting` |
| `model_id`, `fold`, `seed`, `split` | | |
| `slice` | str | `all`, `low_sales`, `high_sales` |
| `metric` | str | `wape`, `wpe`, `mase`, `picp95`, `mpiw95` |
| `value` | float | |
| `n` | int | Number of points scored |
| `calibrated` | bool | Whether intervals were conformalized |

**Slice definition:** `low_sales` is store-product pairs with mean scaled demand below 1, defined as in the dataset authors' repository. Every other pair is `high_sales`. Slice membership is computed once and stored in `slices.parquet` so every consumer uses the same split.


## 3. Change log

|Version|Date|Change|Approved by|
|---|---|---|---|
|0.1|2026-10-28|Initial draft|pending|
