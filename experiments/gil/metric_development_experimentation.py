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
from uncertainty_aware_planner.data.helper_functions import to_imputed_hourly
from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.utils.util_functions import set_seed, get_git_state
from uncertainty_aware_planner.models.imputation import (
    DLinear_Imputation_Quantile,
    TimesNet_Imputation_Quantile,
)
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_percent_errors,
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
TIMESNET_QUANTILE_MEDIAN_CONFIG_FILE = Path(
    "configs/imputation_models/timesnet_imputation_quantile_median.yaml"
)

# dirs
DLINEAR_MODEL_DIRECTORY = Path("experiments/gil/models/seed_86400_dlinear_quantile")
TIMESNET_MODEL_DIRECTORY = Path("experiments/gil/models/seed_86400_timesnet")
REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("experiments/gil/data")

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # loading config files for model
# dlinear_quantile_median_config = load_yaml(DLINEAR_QUANTILE_MEDIAN_CONFIG_FILE)
# # timesnet_quantile_median_config = load_yaml(TIMESNET_QUANTILE_MEDIAN_CONFIG_FILE)

# # load necessary data pipeline elements
# cfg = load_config()
# cube = load_cube(cfg)
# masks = pd.read_parquet("data/processed/eval_masks.parquet")

# # initialize the model itself
# dlinear_model = DLinear_Imputation_Quantile(dlinear_quantile_median_config)
# # timesnet_model = TimesNet_Imputation_Quantile(timesnet_quantile_median_config)

# # load from the files in the proper model directory
# dlinear_model.load(directory=DLINEAR_MODEL_DIRECTORY)
# # timesnet_model.load(directory=TIMESNET_MODEL_DIRECTORY)

# # build the evaluation data from the calibration data
# print("building calibration set")
# calib_set, calib_index = build_pypots_set(cube, cfg, "calib", masks=masks)

# # get the imputation
# print("imputing dlinear point estimate")
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# start.record()
# dlinear_imputation_point_estimate = dlinear_model.impute_point(calib_set)
# end.record()
# torch.cuda.synchronize()
# print(f"{start.elapsed_time(end):.2f} ms")
# print("imputing timesnet point estimate")
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# start.record()
# timesnet_imputation_point_estimate = timesnet_model.impute_point(calib_set)
# end.record()
# torch.cuda.synchronize()
# print(f"{start.elapsed_time(end):.2f} ms")


# def timed(fn, *args, **kwargs):
#     """Run fn and return (result, elapsed_ms),
#     using CUDA events on GPU and perf_counter on CPU."""
#     if torch.cuda.is_available():
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         torch.cuda.synchronize()
#         start.record()
#         result = fn(*args, **kwargs)
#         end.record()
#         torch.cuda.synchronize()
#         return result, start.elapsed_time(end)
#     t0 = time.perf_counter()
#     result = fn(*args, **kwargs)
#     return result, (time.perf_counter() - t0) * 1000


# print("imputing all three quantiles")
# imputation, ms = timed(dlinear_model.impute, calib_set)
# print(f"{ms:.2f} ms")

# print("constructing output")
# lower_quantile_df = to_imputed_hourly(calib_set, calib_index, imputation[..., 0, 0])
# median_df = to_imputed_hourly(calib_set, calib_index, imputation[..., 0, 1])
# upper_quantile_df = to_imputed_hourly(calib_set, calib_index, imputation[..., 0, 2])

# output_df = median_df[
#     ["store_id", "product_id", "datetime", "actual_demand", "is_stockout", "is_masked"]
# ].copy()
# output_df["q025"] = lower_quantile_df["estimated_demand"].copy()
# output_df["estimated_demand"] = median_df["estimated_demand"].copy()
# output_df["q975"] = upper_quantile_df["estimated_demand"].copy()
# output_df.to_parquet(
#     path=(DATA_DIR / "dlinear_calibration_set_imputation.parquet"), index=False
# )

# # get the WAPE/WPE
# print("calculating dlinear WAPE and WPE")
# start = time.perf_counter()
# dlinear_result = calculate_imputation_percent_errors(
#     eval_pypots=calib_set,
#     imputation_point_estimate=dlinear_imputation_point_estimate,
#     daily_pool=False,
# )
# elapsed = time.perf_counter() - start
# print(f"{elapsed:.4f} s")

# print("calculating timesnet WAPE and WPE")
# start = time.perf_counter()
# timesnet_result = calculate_imputation_percent_errors(
#     eval_pypots=calib_set,
#     imputation_point_estimate=timesnet_imputation_point_estimate,
#     daily_pool=False,
# )
# elapsed = time.perf_counter() - start
# print(f"{elapsed:.4f} s")

# print("dlinear: ")
# for key, value in dlinear_result.items():
#     print(key)
#     print(value)

# print("timesnet: ")
# for key, value in timesnet_result.items():
#     print(key)
#     print(value)

# # load imputed calibration data
output_df = pd.read_parquet(
    path=(DATA_DIR / "dlinear_calibration_set_imputation.parquet")
)

empirical_coverage = prediction_interval_coverage_fraction(output_df)
print("empirical coverage: ", empirical_coverage)
mean_width_actuals = mean_prediction_interval_width(output_df)
mean_width_all = mean_prediction_interval_width(output_df, include_stockout=True)
print("mean interval width, masked only: ", mean_width_actuals)
print("mean interval width, all intervals: ", mean_width_all)
