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
    TimesNet_Imputation_Quantile
)
from uncertainty_aware_planner.evaluation.error_metrics import calculate_imputation_percent_errors

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

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# loading config files for model
dlinear_quantile_median_config = load_yaml(DLINEAR_QUANTILE_MEDIAN_CONFIG_FILE)
timesnet_quantile_median_config = load_yaml(TIMESNET_QUANTILE_MEDIAN_CONFIG_FILE)

# load necessary data pipeline elements
cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")

# initialize the model itself
dlinear_model = DLinear_Imputation_Quantile(dlinear_quantile_median_config)
timesnet_model = TimesNet_Imputation_Quantile(timesnet_quantile_median_config)

# load from the files in the proper model directory
dlinear_model.load(directory = DLINEAR_MODEL_DIRECTORY)
timesnet_model.load(directory = TIMESNET_MODEL_DIRECTORY)

# build the evaluation data from the calibration data
print("building calibration set")
calib_set, calib_index = build_pypots_set(cube, cfg, "calib", masks=masks)


# get the imputation
print("imputing dlinear point estimate")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
dlinear_imputation_point_estimate = dlinear_model.impute_point(calib_set)
end.record()
torch.cuda.synchronize()
print(f"{start.elapsed_time(end):.2f} ms")
print("imputing timesnet point estimate")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
timesnet_imputation_point_estimate = timesnet_model.impute_point(calib_set)
end.record()
torch.cuda.synchronize()
print(f"{start.elapsed_time(end):.2f} ms")

# print("imputing all three quantiles")
# imputation = dlinear_model.impute(calib_set)

# print("constructing output")
# lower_quantile_df = to_imputed_hourly(calib_set, calib_index, imputation[..., 0])
# median_df = to_imputed_hourly(calib_set, calib_index, imputation[..., 1])
# upper_quantile_df = to_imputed_hourly(calib_set, calib_index, imputation[..., 2])

# output_df = median_df[["store_id", "product_id", "datetime", "actual_demand", "is_stockout", "is_masked"]].copy()
# output_df["lower_quantile"] = lower_quantile_df["estimated_demand"].copy()
# output_df["estimated_demand"] = median_df["estimated_demand"].copy()
# output_df["upper_quantile"] = upper_quantile_df["estimated_demand"].copy()
# print(output_df.head())

# get the WAPE/WPE
print("calculating dlinear WAPE and WPE")
start = time.perf_counter()
dlinear_result = calculate_imputation_percent_errors(
    eval_pypots=calib_set, 
    imputation_point_estimate=dlinear_imputation_point_estimate, 
    daily_pool=False
)
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f} s")

print("calculating timesnet WAPE and WPE")
start = time.perf_counter()
timesnet_result = calculate_imputation_percent_errors(
    eval_pypots=calib_set, 
    imputation_point_estimate=timesnet_imputation_point_estimate, 
    daily_pool=False
)
elapsed = time.perf_counter() - start
print(f"{elapsed:.4f} s")

print("dlinear: ")
for key, value in dlinear_result.item():
    print(key)
    print(value)

print("timesnet: ")
for key, value in timesnet_result.item():
    print(key)
    print(value)