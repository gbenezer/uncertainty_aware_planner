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
from uncertainty_aware_planner.data.helper_functions import (
    to_imputed_hourly,
    to_imputed_hourly_interval,
    to_imputed_hourly_interval_median,
    get_series_hourly,
    add_series_metadata,
    filter_by_mean_demand,
)
from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.utils.util_functions import set_seed, get_git_state
from uncertainty_aware_planner.models.imputation import (
    DLinear_Imputation_Quantile,
    DLinear_Imputation_Baseline,
    TimesNet_Imputation_Quantile,
)
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_errors_df,
    mean_absolute_scaled_error,
)
from uncertainty_aware_planner.evaluation.uncertainty_metrics import (
    prediction_interval_coverage_fraction,
    mean_prediction_interval_width,
)

# script constants

# model configs
DLINEAR_BASELINE_CONFIG_FILE = Path(
    "configs/imputation_models/dlinear_imputation_baseline.yaml"
)
DLINEAR_QUANTILE_CONFIG_FILE = Path(
    "configs/imputation_models/dlinear_imputation_quantile.yaml"
)
DLINEAR_QUANTILE_MEDIAN_CONFIG_FILE = Path(
    "configs/imputation_models/dlinear_imputation_quantile_median.yaml"
)
TIMESNET_QUANTILE_MEDIAN_CONFIG_FILE = Path(
    "configs/imputation_models/timesnet_imputation_quantile_median.yaml"
)

# data configs
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")

# dirs
DATA_DIR = Path("experiments/gil/data")
OUTPUT_DATA_DIRECTORY = Path("results/imputation")
MODEL_DIRECTORY = Path("experiments/gil/models")
RESULTS_FILE = (
    OUTPUT_DATA_DIRECTORY
    / "full_train_data_calbration_data_eval_imputation_no_conformalization.csv"
)
REPO_DIR = Path(__file__).resolve().parent

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEEDS = [86400, 4, 42]

# Get a working imputation baseline experiment log commit file hash
SHA = get_git_state(REPO_DIR)

# Get the appropriate hourly series data once
cache = DATA_DIR / "series_hourly.parquet"
if cache.exists():
    SERIES_HOURLY_DF = pd.read_parquet(cache)
else:
    SERIES_HOURLY_DF = get_series_hourly()
    SERIES_HOURLY_DF.to_parquet(cache)

# resume: figure out which runs are already done
if RESULTS_FILE.exists():
    done = pd.read_csv(
        RESULTS_FILE, usecols=["seed", "model_type", "num_quantiles"]
    ).drop_duplicates()
    completed = set(zip(done["seed"], done["model_type"], done["num_quantiles"]))
else:
    completed = set()

# loading config files for models
dlinear_baseline_config = load_yaml(DLINEAR_BASELINE_CONFIG_FILE)
dlinear_quantile_config = load_yaml(DLINEAR_QUANTILE_CONFIG_FILE)
dlinear_quantile_median_config = load_yaml(DLINEAR_QUANTILE_MEDIAN_CONFIG_FILE)
timesnet_quantile_median_config = load_yaml(TIMESNET_QUANTILE_MEDIAN_CONFIG_FILE)
CONFIGS = [
    dlinear_quantile_config,
    dlinear_quantile_median_config,
    timesnet_quantile_median_config,
    dlinear_baseline_config,
]

# load necessary data pipeline elements
cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")

# constructing the experiment-wide metadata dictionary
experiment_dict = {}
experiment_dict["repo_dir_commit"] = SHA
experiment_dict["public_data_config_file"] = PUBLIC_DATA_CONFIG_FILE
experiment_dict["split_config_file"] = SPLIT_CONFIG_FILE
experiment_dict["device"] = DEVICE

for seed in SEEDS:
    for config in CONFIGS:
        # set the seed under each config repetition
        set_seed(seed)

        # initialize the run results list
        run_results = []

        # accessing the type of model and quantile information
        model_type = config["name"]
        model_quantiles = config.get("quantiles")
        model_num_quantiles = (
            len(model_quantiles) if model_quantiles is not None else None
        )

        # checking for completion
        if (seed, model_type, model_num_quantiles) in completed:
            print(
                f"Skipping seed={seed}, model_type={model_type}, num_quantiles={model_num_quantiles} (already done)"
            )
            continue

        # defining the per-run dictionary after assessing necessity
        run_dict = {}
        run_dict["execution_datetime"] = datetime.datetime.now()
        run_dict["seed"] = seed
        run_dict["eval_data"] = "calibration"
        run_dict["model_type"] = model_type
        run_dict["quantiles"] = model_quantiles
        run_dict["num_quantiles"] = model_num_quantiles

        # initialize the training set
        calib_set, calib_index = build_pypots_set(cube, cfg, "calib", masks=masks)

        # initialize the model subdirectory prefix
        model_subdirectory_prefix = f"seed_{seed}_"

        # initializing the proper model
        if model_type == "DLinear_Imputation_Baseline":
            load_directory = MODEL_DIRECTORY / (
                model_subdirectory_prefix + "dlinear_baseline"
            )
            imputation_model = DLinear_Imputation_Baseline(config)
            imputation_model.load(load_directory)

        elif model_type == "DLinear_Imputation_Quantile":
            imputation_model = DLinear_Imputation_Quantile(config)

            if model_num_quantiles == 3:
                load_directory = MODEL_DIRECTORY / (
                    model_subdirectory_prefix + "dlinear_quantile"
                )
                imputation_model.load(load_directory)

            elif model_num_quantiles == 2:
                load_directory = MODEL_DIRECTORY / (
                    model_subdirectory_prefix + "dlinear_quantile_interval_only"
                )
                imputation_model.load(load_directory)

            else:
                print("The imputation model remains unloaded")

        else:
            assert model_type == "TimesNet_Imputation_Quantile"
            imputation_model = TimesNet_Imputation_Quantile(config)
            load_directory = MODEL_DIRECTORY / (model_subdirectory_prefix + "timesnet")
            imputation_model.load(load_directory)

        # impute the data
        imputation_output = imputation_model.impute(calib_set)

        # call the proper function for conversion to a dataframe
        # (leaving clipping on)
        if model_type == "DLinear_Imputation_Baseline":
            imputation_df = to_imputed_hourly(
                calib_set, calib_index, imputation_output[..., 0]
            )
            imputation_df = imputation_df.assign(q025=np.nan, q975=np.nan)

        elif model_num_quantiles == 2:
            imputation_df = to_imputed_hourly_interval(
                calib_set, calib_index, imputation_output
            )

        elif model_num_quantiles == 3:
            imputation_df = to_imputed_hourly_interval_median(
                calib_set, calib_index, imputation_output
            )

        # add back the metadata to the imputation dataframe
        imputation_df = add_series_metadata()

        # evaluating the imputation overall
        overall_pe = calculate_imputation_errors_df(imputed_hourly=imputation_df)
        overall_1hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_df, lag_hours=1, series_hourly=SERIES_HOURLY_DF
        )
        overall_24hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_df, lag_hours=24, series_hourly=SERIES_HOURLY_DF
        )
        overall_168hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_df, lag_hours=168, series_hourly=SERIES_HOURLY_DF
        )

        # evaluating the uncertainty estimation
        if model_num_quantiles is None:
            # no values for baseline
            overall_picp = {"value": np.nan, "n_picp": 0}
            overall_mpiw = {"value": np.nan, "n_mpiw": 0}

        else:
            overall_picp = prediction_interval_coverage_fraction(
                actual_and_interval_df=imputation_df
            )
            overall_mpiw = mean_prediction_interval_width(interval_df=imputation_df)

        # calculating the metrics disaggregated on weekday vs. holiday + weekend
        imputation_weekday = imputation_df[
            imputation_df["weekday"] & ~imputation_df["holiday_flag"]
        ]
        imputation_weekend = imputation_df[
            ~imputation_df["weekday"] | imputation_df["holiday_flag"]
        ]

        # evaluating the imputation on the weekdays
        weekday_pe = calculate_imputation_errors_df(imputed_hourly=imputation_weekday)
        weekday_1hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_weekday, lag_hours=1, series_hourly=SERIES_HOURLY_DF
        )
        weekday_24hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_weekday, lag_hours=24, series_hourly=SERIES_HOURLY_DF
        )
        weekday_168hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_weekday, lag_hours=168, series_hourly=SERIES_HOURLY_DF
        )

        # evaluating the uncertainty estimation
        if model_num_quantiles is None:
            # no values for baseline
            weekday_picp = {"value": np.nan, "n_picp": 0}
            weekday_mpiw = {"value": np.nan, "n_mpiw": 0}

        else:
            weekday_picp = prediction_interval_coverage_fraction(
                actual_and_interval_df=imputation_weekday
            )
            weekday_mpiw = mean_prediction_interval_width(
                interval_df=imputation_weekday
            )

        # evaluating the imputation on the weekends
        weekend_pe = calculate_imputation_errors_df(imputed_hourly=imputation_weekend)
        weekend_1hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_weekend, lag_hours=1, series_hourly=SERIES_HOURLY_DF
        )
        weekend_24hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_weekend, lag_hours=24, series_hourly=SERIES_HOURLY_DF
        )
        weekend_168hr_mase = mean_absolute_scaled_error(
            imputed_df=imputation_weekend, lag_hours=168, series_hourly=SERIES_HOURLY_DF
        )

        # evaluating the uncertainty estimation
        if model_num_quantiles is None:
            # no values for baseline
            weekend_picp = {"value": np.nan, "n_picp": 0}
            weekend_mpiw = {"value": np.nan, "n_mpiw": 0}

        else:
            weekend_picp = prediction_interval_coverage_fraction(
                actual_and_interval_df=imputation_weekend
            )
            weekend_mpiw = mean_prediction_interval_width(
                interval_df=imputation_weekend
            )

        # disaggregating further on absolute low vs. high sales volume (hourly mean, across entire dataset)
        imputation_low_volume = filter_by_mean_demand(
            imputed_df=imputation_df,
            series_hourly=SERIES_HOURLY_DF,
            group="low",
            mean_source="full",
        )
        imputation_high_volume = filter_by_mean_demand(
            imputed_df=imputation_df,
            series_hourly=SERIES_HOURLY_DF,
            group="high",
            mean_source="full",
        )
        imputation_weekday_low_volume = filter_by_mean_demand(
            imputed_df=imputation_weekday,
            series_hourly=SERIES_HOURLY_DF,
            group="low",
            mean_source="full",
        )
        imputation_weekday_high_volume = filter_by_mean_demand(
            imputed_df=imputation_weekday,
            series_hourly=SERIES_HOURLY_DF,
            group="high",
            mean_source="full",
        )
        imputation_weekend_low_volume = filter_by_mean_demand(
            imputed_df=imputation_weekend,
            series_hourly=SERIES_HOURLY_DF,
            group="low",
            mean_source="full",
        )
        imputation_weekend_high_volume = filter_by_mean_demand(
            imputed_df=imputation_weekend,
            series_hourly=SERIES_HOURLY_DF,
            group="high",
            mean_source="full",
        )

    #         # update with metadata and add to the run results
    #         result.update({**experiment_dict, **run_dict})
    #         run_results.append(result)

    #     # one append per run, so a run is either fully recorded or not at all
    #     pd.DataFrame(run_results).to_csv(
    #         RESULTS_FILE,
    #         mode="a",
    #         header=not RESULTS_FILE.exists(),
    #         index=False,
    #     )
