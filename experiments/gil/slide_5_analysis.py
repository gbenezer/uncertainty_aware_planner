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
SEEDS = [86400, 4, 42]
THRESHOLD = 1 / 24
MEAN_SOURCE = "full"
MODEL_TYPE_QUANTILE_NUMBERS = [
    ("DLinear_Imputation_Baseline", None),
    ("DLinear_Imputation_Quantile", 2),
    ("DLinear_Imputation_Quantile", 3),
    ("TimesNet_Imputation_Quantile", 3),
]
SERIES_HOURLY = ieh.load_series_hourly("data/hf_cache")
UNCERTAINTY_COLUMNS = [
    "store_id",
    "product_id",
    "picp",
    "coverage_gap",
    "n_picp",
    "mpiw",
    "median_piw",
    "std_piw",
    "n_mpiw",
]

for model_combo in MODEL_TYPE_QUANTILE_NUMBERS:
    # extract relevant model parameters
    model_type = model_combo[0]
    num_quantiles = model_combo[1]
    calculate_uncertainty = num_quantiles is not None

    # output the model parameters to stdout
    print(f"Model Type: {model_type}")
    print(f"Number of Quantiles: {num_quantiles if calculate_uncertainty else 1}")

    for seed in SEEDS:
        print(f"Seed: {seed}")
        file_name = (
            FILE_STEM + f"_seed_{seed}_{model_type}_{num_quantiles}_quantiles.csv"
        )
        file_path = INFERENCE_DATA_DIR / file_name

        # load the appropriate data
        inference_df = help_fun.import_inference_table(file_path)

        # calculate the scaled absolute error
        error_df = em.scaled_absolute_errors(
            imputed_df=inference_df, lags=(1,), series_hourly=SERIES_HOURLY
        )

        # add the inference data to the calculated errors
        error_eval_df = inference_df.join(error_df)

        # slice along the low and high volume partitions
        low_volume_error_eval_df = help_fun.filter_by_mean_demand(
            error_eval_df,
            SERIES_HOURLY,
            group="low",
            threshold=THRESHOLD,
            mean_source=MEAN_SOURCE,
        )
        high_volume_error_eval_df = help_fun.filter_by_mean_demand(
            error_eval_df,
            SERIES_HOURLY,
            group="high",
            threshold=THRESHOLD,
            mean_source=MEAN_SOURCE,
        )

        # summarize the 1-hour MASE across series
        store_product_mase_summary = (
            error_eval_df[error_eval_df["evaluated"]]
            .groupby(["store_id", "product_id"])
            .agg({"ase_1hr": "mean"})
            .reset_index()
            .sort_values(by="ase_1hr", ascending=False)
        )
        low_vol_store_product_mase_summary = (
            low_volume_error_eval_df[low_volume_error_eval_df["evaluated"]]
            .groupby(["store_id", "product_id"])
            .agg({"ase_1hr": "mean"})
            .reset_index()
            .sort_values(by="ase_1hr", ascending=False)
        )
        high_vol_store_product_mase_summary = (
            high_volume_error_eval_df[high_volume_error_eval_df["evaluated"]]
            .groupby(["store_id", "product_id"])
            .agg({"ase_1hr": "mean"})
            .reset_index()
            .sort_values(by="ase_1hr", ascending=False)
        )

        print("\n")

        print("Overall Top 5 1-hour MASE Store-Product Series")
        print(store_product_mase_summary.head())
        print("Low Volume Top 5 1-hour MASE Store-Product Series")
        print(low_vol_store_product_mase_summary.head())
        print("High Volume Top 5 1-hour MASE Store-Product Series")
        print(high_vol_store_product_mase_summary.head())

        # the baseline model has no interval columns, so skip uncertainty for it
        if calculate_uncertainty:
            # evaluating uncertainty metrics on a per-series basis
            store_product_uncertainty_summary = um.interval_metrics_by_group(
                error_eval_df
            )
            low_volume_store_product_uncertainty_summary = um.interval_metrics_by_group(
                low_volume_error_eval_df
            )
            high_volume_store_product_uncertainty_summary = (
                um.interval_metrics_by_group(high_volume_error_eval_df)
            )

            # sort by PICP
            store_product_picp = store_product_uncertainty_summary.sort_values(
                by="picp"
            )
            low_volume_store_product_picp = (
                low_volume_store_product_uncertainty_summary.sort_values(by="picp")
            )
            high_volume_store_product_picp = (
                high_volume_store_product_uncertainty_summary.sort_values(by="picp")
            )

            # sort by MPIW
            store_product_mpiw = store_product_uncertainty_summary.sort_values(
                by="mpiw", ascending=False
            )
            low_volume_store_product_mpiw = (
                low_volume_store_product_uncertainty_summary.sort_values(
                    by="mpiw", ascending=False
                )
            )
            high_volume_store_product_mpiw = (
                high_volume_store_product_uncertainty_summary.sort_values(
                    by="mpiw", ascending=False
                )
            )

            print("\n")

            print("Overall Bottom 5 PICP Store-Product Series")
            print(store_product_picp[UNCERTAINTY_COLUMNS].head())
            print("Low Volume Bottom 5 PICP Store-Product Series")
            print(low_volume_store_product_picp[UNCERTAINTY_COLUMNS].head())
            print("High Volume Bottom 5 PICP Store-Product Series")
            print(high_volume_store_product_picp[UNCERTAINTY_COLUMNS].head())

            print("\n")

            print("Overall Top 5 MPIW Store-Product Series")
            print(store_product_mpiw[UNCERTAINTY_COLUMNS].head())
            print("Low Volume Top 5 MPIW Store-Product Series")
            print(low_volume_store_product_mpiw[UNCERTAINTY_COLUMNS].head())
            print("High Volume Top 5 MPIW Store-Product Series")
            print(high_volume_store_product_mpiw[UNCERTAINTY_COLUMNS].head())

        print("\n")
        print("\n")
        print("\n")
