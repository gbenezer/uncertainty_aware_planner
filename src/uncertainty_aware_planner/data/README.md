# Data stage

Loads FreshRetailNet-50K at a pinned revision, then validates, cleans and splits it. It produces the shared tables in `contracts/CONTRACTS.md` and the training inputs for the imputation and forecasting models. Settings live in `configs/data.yaml` and `configs/splits.yaml`. Outputs go to `data/`, which is gitignored and never committed.

**Order of work:** everyone runs Steps 1–2. Imputation owners do Step 3, and the data lead or the imputation owner runs Step 4 on the chosen imputation run. Forecasting owners do Step 5, which needs a `demand_daily` file from Step 4. Until one exists, see the stand-in at the end of Step 5.

## Step 1. Set up (everyone)

```bash
uv venv --python 3.12.3
uv sync --frozen --all-groups
```

## Step 2. Prepare the shared data (everyone, once per machine)

```bash
uv run python -m uncertainty_aware_planner.data load     # download + validate
uv run python -m uncertainty_aware_planner.data splits   # data/processed/splits.parquet
uv run python -m uncertainty_aware_planner.data masks    # data/processed/eval_masks.parquet
```

- **`load`:** the first run downloads about 2.4 GB into `data/hf_cache/`. After that, a load takes about 15 s and 4.5 GB of memory. It must print `50000 series x 97 days (2024-03-28 to 2024-07-02) passed validation`. It also writes `data/raw/DATA_VERSION.json`.
- **`splits`:** prints the fold and final windows.
- **`masks`:** writes the in-stock hours hidden for imputation scoring.
- **Same results on every machine:** outputs depend only on the pinned data and the committed configs. Don't pass `--seed`, or your masks will differ from everyone else's.

Every output gets a `.manifest.json` (config, seed, dataset revision, counts) for the run log.

## Step 3. Imputation models (PyPOTS: DLinear, TimesNet)

**3.1 Load the data and conversion helper function into your model file.** The helper function `to_imputed_hourly` turns a PyPOTS prediction into `imputed_hourly` rows. It writes one row per hour the model filled, and the index's `keep_start` stops overlapping windows from filling an hour twice.

```python
from pathlib import Path

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.helper_functions import to_imputed_hourly

cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")
```

**3.2 Cross-validation (folds 1–6).** Train on each fold's `train` window and predict its `val` window. The `val` set has the shared masks applied: the masked hours are NaN in `X`, and their true sales are in `X_ori`.

```python
for fold in range(1, 7):
    train_set, _ = build_pypots_set(cube, cfg, "train", fold=fold)  # {"X"}
    val_set, val_index = build_pypots_set(
        cube, cfg, "val", fold=fold, masks=masks
    )  # {"X", "X_ori"}
    model = ...  # e.g. pypots.imputation.DLinear(n_steps=672, n_features=6, ...); settings are yours
    model.fit(train_set)
    rows = to_imputed_hourly(
        val_set, val_index, model.predict(val_set)["imputation"][..., 0]
    )
    # rows with a non-null actual_demand are the masked hours evaluation scores on
```

**3.3 Final model.** Train on 03-28..06-18 only (the calibration week is held out), then predict the calibration week. Its masked hours are what conformal calibration and the held-out check use.

```python
train_set, _ = build_pypots_set(cube, cfg, "train")  # (150000, 672, 6), 2.4 GB
model = ...  # a new model with the settings you chose in 3.2
model.fit(train_set)
calib_set, calib_index = build_pypots_set(cube, cfg, "calib", masks=masks)
calib_rows = to_imputed_hourly(
    calib_set, calib_index, model.predict(calib_set)["imputation"][..., 0]
)
```

**3.4 Fill every real stockout and save the file for Step 4.** The `recovery` part covers 03-28..06-25, so every real stockout hour gets exactly one row, all with a null `actual_demand`.

```python
recovery_set, recovery_index = build_pypots_set(
    cube, cfg, "recovery"
)  # (200000, 672, 6), 3.2 GB
rows = to_imputed_hourly(
    recovery_set, recovery_index, model.predict(recovery_set)["imputation"][..., 0]
)
output = Path(
    "data/processed/imputed_hourly/imp-dlinear-quantile-1024.parquet"
)  # <model_id>-<seed>
output.parent.mkdir(parents=True, exist_ok=True)
rows.to_parquet(output)
```

- **Inputs:** each sample is one 28-day window of one series, 672 hourly steps × 6 features: `[sales, discount, holiday_flag, precpt, avg_temperature, hour/23]`. Sales are feature 0 and are what you impute. Real stockout hours are NaN.
- **Columns:** `to_imputed_hourly` gives the five columns Step 4 reads. Before handing the tables to evaluation, add the rest of CONTRACTS.md §2.3: model id, seed, fold, split and quantiles.
- **Memory:** the final arrays come on top of the 4.5 GB load. Build one part at a time and `del` it when you're done.

## Step 4. Build `demand_daily` from one imputation run

```bash
uv run python -m uncertainty_aware_planner.data daily \
    --imputed data/processed/imputed_hourly/imp-dlinear-quantile-1024.parquet \
    --run-id imp-dlinear-quantile-1024
```

- **What it builds:** `data/processed/demand_daily/<run-id>.parquet`, which is observed in-stock sales plus the imputed stockout hours, summed per day. That is what forecasting trains on.
- **No overwrites:** the file is never overwritten, so every new imputation run needs a new run ID.
- **Checks:** it stops with an error if a row fills an hour that wasn't a stockout, if an hour appears twice, or if a day is only partly filled. Pass the Step 3.4 `recovery` file. Masked rows (non-null `actual_demand`) are ignored.
- **Which run to use:** the Feature Cards say forecasting uses the top-performing imputation model. Until the team picks one, use the baseline run.
- **Planner edits:** add `--edit-log PATH` to apply the CONTRACTS.md §2.9 `edit_log`. The applied `edit_id`s go into the manifest.

## Step 5. Forecasting models (pytorch-forecasting: TFT, DLinear)

**5.1 Cross-validation (folds 1–6).** Each fold trains on one 21-day input plus a 7-day forecast per series, then forecasts 7 days at a time across that fold's 4-week `val` window.

```python
import pandas as pd

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.timeseries_dataset import build_timeseries_datasets

cfg = load_config()
demand = pd.read_parquet(
    "data/processed/demand_daily/imp-dlinear-quantile-1024.parquet"
)

for fold in range(1, 7):
    train_ds, val_ds, spec = build_timeseries_datasets(demand, cfg, fold=fold)
    train_dl = train_ds.to_dataloader(train=True, batch_size=..., num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=..., num_workers=0)
    model = (
        ...
    )  # e.g. TemporalFusionTransformer.from_dataset(train_ds, ...); settings are yours
```

**5.2 Final model.** Train on 03-28..06-18 with a 70-day input. Forecast the calibration week for conformal calibration, then forecast the test week **once**, at the end.

```python
train_ds, calib_ds, spec = build_timeseries_datasets(demand, cfg, eval_part="calib")
train_ds, test_ds, spec = build_timeseries_datasets(demand, cfg, eval_part="test")
```

- **Target and features:** the target is `demand_total`. Field roles follow the dataset authors' TFT config. Scalers and encoders are fit on the training window only.
- **Test week:** test-week targets are set to 0 inside `test_ds`, so the model never sees them. Evaluation reads the actuals from `demand_daily`.
- **Output:** write predictions in the CONTRACTS.md §2.5 `forecast` format.
- **`spec`:** the exact dataset settings, as JSON for the run log.

**Stand-in before the first imputation run (fixtures only, placeholder values).** This fills every stockout with 0.5, so the numbers mean nothing. Use it only to get your model code running.

```python
# run with PYTHONPATH=src:.
import tempfile
from datetime import date

from fixtures.synthetic_raw_series import fixture_config
from tests.data.helpers import fake_imputed_hourly
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.daily import build_demand_daily

cfg = fixture_config(tempfile.mkdtemp())
cube = load_cube(cfg)
imputed = fake_imputed_hourly(cube, date(2024, 3, 28), date(2024, 6, 25))
demand, _ = build_demand_daily(cube, imputed, "fixture-stand-in", cfg["splits"])
```

## Things the team should know

- **Data length:** training is 90 days (2024-03-28..06-25) and the official test is 06-26..07-02. It is not "14 weeks".
- **Input lengths:** each fold window holds one 21-day input plus a 7-day forecast, so each series gives **one training sample per fold**. The final run uses a 70-day input (the authors' TFT; their DLinear used 62). Both are set in `splits.yaml`, and the team has to agree on one shared value.
- **Masking:** imputation covers all 24 hours, as the Feature Cards say. Masks only hit fully in-stock days, which is 40.0% of training days over 24 hours, versus 55.7% over hours 6–21.
- **Discount cleaning:** `discount` 0.0 (17,339 rows) and values above 1 (34 rows) become 1.0. Nothing else is changed.
