from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import datetime

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.helper_functions import to_imputed_hourly, to_imputed_hourly_interval_median, get_series_hourly
from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.utils.util_functions import set_seed, get_git_state
from uncertainty_aware_planner.models.imputation import (
    DLinear_Imputation_Quantile,
)
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_percent_errors,
    calculate_imputation_errors_df,
    mean_absolute_scaled_error
)
from uncertainty_aware_planner.evaluation.uncertainty_metrics import (
    prediction_interval_coverage_fraction,
    mean_prediction_interval_width,
)

# script constants

# model config
DLINEAR_QUANTILE_MEDIAN_CONFIG_FILE = Path(
    "configs/imputation_models/dlinear_imputation_quantile_median.yaml"
)

# dirs
DLINEAR_MODEL_DIRECTORY = Path("experiments/gil/models/seed_86400_dlinear_quantile")
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("experiments/gil/data")

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# loading config files for model
dlinear_quantile_median_config = load_yaml(DLINEAR_QUANTILE_MEDIAN_CONFIG_FILE)

# load necessary data pipeline elements
cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")

# # initialize the model itself
dlinear_model = DLinear_Imputation_Quantile(dlinear_quantile_median_config)

# load from the files in the proper model directory
dlinear_model.load(directory=DLINEAR_MODEL_DIRECTORY)

# build the evaluation data from the calibration data
print("building calibration set")
calib_set, calib_index = build_pypots_set(cube, cfg, "calib", masks=masks)
calibration_df = to_imputed_hourly(calib_set, calib_index, calib_set["X_ori"][..., 0])

def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms),
    using CUDA events on GPU and perf_counter on CPU."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


print("imputing all three quantiles")
imputation, ms = timed(dlinear_model.impute, calib_set)
print(f"{ms:.2f} ms")

# extracting the appropriate point estimate
point_estimate = imputation[..., 0, 1]

print("constructing dfs")

df_point = to_imputed_hourly(calib_set, calib_index, point_estimate)
df_point_raw = to_imputed_hourly(calib_set, calib_index, point_estimate, clip_demand = False)
df_interval = to_imputed_hourly_interval_median(calib_set, calib_index, imputation)
df_interval_raw = to_imputed_hourly_interval_median(calib_set, calib_index, imputation, clip_demand = False)

# get the WAPE/WPE
print("calculating dlinear WAPE and WPE with old function")
start = time.perf_counter()
dlinear_result = calculate_imputation_percent_errors(
    eval_pypots=calib_set,
    imputation_point_estimate=point_estimate,
)
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f} s")
print("Old function, dlinear: ", dlinear_result)

print("calculating dlinear WAPE and WPE with new function and clipping demand")
start = time.perf_counter()
dlinear_result_new = calculate_imputation_errors_df(
    imputed_hourly = df_point
)
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f} s")
print("New function, dlinear, with clipping: ", dlinear_result_new)

print("calculating dlinear WAPE and WPE with new function, no clipping")
start = time.perf_counter()
dlinear_result_new_raw = calculate_imputation_errors_df(
    imputed_hourly = df_interval_raw
)
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f} s")
print("New function, dlinear, no clipping: ", dlinear_result_new_raw)

print("with clipping")
empirical_coverage = prediction_interval_coverage_fraction(df_interval)
print("empirical coverage: ", empirical_coverage)
mean_width_actuals = mean_prediction_interval_width(df_interval)
mean_width_all = mean_prediction_interval_width(df_interval, include_stockout=True)
print("mean interval width, masked only: ", mean_width_actuals)
print("mean interval width, all intervals: ", mean_width_all)

print("without clipping")
empirical_coverage = prediction_interval_coverage_fraction(df_interval_raw)
print("empirical coverage: ", empirical_coverage)
mean_width_actuals = mean_prediction_interval_width(df_interval_raw)
mean_width_all = mean_prediction_interval_width(df_interval_raw, include_stockout=True)
print("mean interval width, masked only: ", mean_width_actuals)
print("mean interval width, all intervals: ", mean_width_all)

print("building full true `series_hourly`")
start = time.perf_counter()
series_hourly = get_series_hourly()
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f} s")
print("calculating MASE, 24 hour lag, clipped")
mase_result = mean_absolute_scaled_error(imputed_df = df_interval, series_hourly=series_hourly)
print(mase_result)
print("calculating MASE, 1 hour lag, clipped")
mase_result = mean_absolute_scaled_error(imputed_df = df_interval, series_hourly=series_hourly, lag_hours=1)
print(mase_result)
print("calculating MASE, 1 week lag, clipped")
mase_result = mean_absolute_scaled_error(imputed_df = df_interval, series_hourly=series_hourly, lag_hours=168)
print(mase_result)


print("calculating MASE, 24 hour lag, raw")
mase_result = mean_absolute_scaled_error(imputed_df = df_interval_raw, series_hourly=series_hourly)
print(mase_result)
print("calculating MASE, 1 hour lag, raw")
mase_result = mean_absolute_scaled_error(imputed_df = df_interval_raw, series_hourly=series_hourly, lag_hours=1)
print(mase_result)
print("calculating MASE, 1 week lag, raw")
mase_result = mean_absolute_scaled_error(imputed_df = df_interval_raw, series_hourly=series_hourly, lag_hours=168)
print(mase_result)