from pathlib import Path
from typing import Union
import pandas as pd
import uncertainty_aware_planner.data.helper_functions as help_fun
import uncertainty_aware_planner.evaluation.imputation_evaluation_helpers as ieh
import uncertainty_aware_planner.evaluation.error_metrics as em
import uncertainty_aware_planner.evaluation.uncertainty_metrics as um

# constants that can be modulated
INFERENCE_DATA_DIR = Path("results/imputation/pretrained_model_eval/model_outputs")
FILE_STEM = "pretrained_model_inference_output"
SEED = 86400
MODEL_TYPE = "DLinear_Imputation_Quantile"
NUM_QUANTILES = 3
EXAMPLE_FILE = FILE_STEM + f"_seed_{SEED}_{MODEL_TYPE}_{NUM_QUANTILES}_quantiles.csv"
EXAMPLE_FILE_PATH = INFERENCE_DATA_DIR / EXAMPLE_FILE

# load the inference df and metadata
test_df = help_fun.import_inference_table(EXAMPLE_FILE_PATH)
series_hourly = ieh.load_series_hourly("data/hf_cache")
test_df_with_data = help_fun.add_series_metadata(
    imputed=test_df, series_hourly=series_hourly
)

# calculate the scaled error for each hour where it is defined
scaled_error_output_df = em.scaled_absolute_errors(
    imputed_df=test_df, lags=(1,), series_hourly=series_hourly
)

# joining the dataframes
eval_df = test_df_with_data.join(scaled_error_output_df)

# getting the low-volume slice
# defined to be 1/24 of a unit per-hour mean demand (1 unit per-day mean demand)
# with the mean taken across the full dataset (train, calibration, and test)
low_volume_eval_df = help_fun.filter_by_mean_demand(
    eval_df, series_hourly, group="low", threshold=1 / 24, mean_source="full"
)

# getting the high-volume slice
# defined to be 1/24 of a unit per-hour mean demand (1 unit per-day mean demand)
# with the mean taken across the full dataset (train, calibration, and test)
high_volume_eval_df = help_fun.filter_by_mean_demand(
    eval_df, series_hourly, group="high", threshold=1 / 24, mean_source="full"
)


# getting the store-product series that have the highest 1-hour MASE
# with the scale defined over the calibration week
store_product_mase_summary = (
    eval_df[eval_df["evaluated"]]
    .groupby(["store_id", "product_id"])
    .agg({"ase_1hr": "mean"})
    .reset_index()
    .sort_values(by="ase_1hr", ascending=False)
)
low_vol_store_product_mase_summary = (
    low_volume_eval_df[low_volume_eval_df["evaluated"]]
    .groupby(["store_id", "product_id"])
    .agg({"ase_1hr": "mean"})
    .reset_index()
    .sort_values(by="ase_1hr", ascending=False)
)

high_vol_store_product_mase_summary = (
    high_volume_eval_df[high_volume_eval_df["evaluated"]]
    .groupby(["store_id", "product_id"])
    .agg({"ase_1hr": "mean"})
    .reset_index()
    .sort_values(by="ase_1hr", ascending=False)
)

print("Overall Top 5 1-hour MASE Store-Product Series")
print(store_product_mase_summary.head())
print("Low Volume Top 5 1-hour MASE Store-Product Series")
print(low_vol_store_product_mase_summary.head())
print("High Volume Top 5 1-hour MASE Store-Product Series")
print(high_vol_store_product_mase_summary.head())

# evaluating uncertainty metrics on a per-series basis
store_product_uncertainty_summary = um.interval_metrics_by_group(eval_df)
low_volume_store_product_uncertainty_summary = um.interval_metrics_by_group(
    low_volume_eval_df
)
high_volume_store_product_uncertainty_summary = um.interval_metrics_by_group(
    high_volume_eval_df
)

print("Overall Bottom 5 PICP Store-Product Series")
print(store_product_uncertainty_summary.sort_values(by="picp").head())
print("Low Volume Bottom 5 PICP Store-Product Series")
print(low_volume_store_product_uncertainty_summary.sort_values(by="picp").head())
print("High Volume Bottom 5 PICP Store-Product Series")
print(high_vol_store_product_mase_summary.sort_values(by="picp").head())

print("Overall Top 5 MPIW Store-Product Series")
print(store_product_uncertainty_summary.sort_values(by="mpiw", ascending=False).head())
print("Low Volume Top 5 MPIW Store-Product Series")
print(
    low_volume_store_product_uncertainty_summary.sort_values(
        by="mpiw", ascending=False
    ).head()
)
print("High Volume Top 5 MPIW Store-Product Series")
print(
    high_volume_store_product_uncertainty_summary.sort_values(
        by="mpiw", ascending=False
    ).head()
)
