from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
import datetime

from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.helper_functions import to_imputed_hourly
from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_percent_errors,
)
from uncertainty_aware_planner.utils.util_functions import set_seed

# script constants
FOLD = 1
DLINEAR_CONFIG_FILE = Path("configs/imputation_models/dlinear_imputation_baseline.yaml")
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")
EXPERIMENT_DATA_DIRECTORY = Path("experiments/gil/data")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEEDS = [42, 2048, 1]

# temporary helper function to get to a working imputation baseline experiment log
def get_git_hash(repo_dir=None) -> str:
    repo_dir = repo_dir or Path(__file__).resolve().parent
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir
    ).decode().strip()

# get the hash
try:
    dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
    SHA = get_git_hash() + ("-dirty" if dirty else "")
except (subprocess.CalledProcessError, FileNotFoundError):
    SHA = "unknown"


# loading config file for baseline model
dlinear_config = load_yaml(DLINEAR_CONFIG_FILE)

# load necessary data pipeline elements
cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")

# constructing the experiment-wide metadata dictionary
experiment_dict = {}
experiment_dict["commit"] = SHA
experiment_dict["hyperparameter_config_file"] = DLINEAR_CONFIG_FILE
experiment_dict["public_data_config_file"] = PUBLIC_DATA_CONFIG_FILE
experiment_dict["split_config_file"] = SPLIT_CONFIG_FILE
experiment_dict["device"] = DEVICE

# initialize an empty list to hold results
result_dict_list = []

for seed in SEEDS:
    for fold in range(1, 7):

        # defining the per-run dictionary
        run_dict = {}
        run_dict["execution_datetime"] = datetime.datetime.now()
        run_dict["seed"] = seed
        run_dict["fold"] = fold

        # set the seed
        set_seed(seed)

        # retrieve the correct imputation train and validation data
        train_set, _ = build_pypots_set(cube, cfg, "train", fold=fold)
        val_set, val_index = build_pypots_set(cube, cfg, "val", fold=fold, masks=masks)

        # initialize the baseline model with the correct hyperparameters
        model = DLinear(
            n_steps = dlinear_config["n_steps"],
            n_features = dlinear_config["n_features"],
            moving_avg_window_size = dlinear_config["moving_avg_window_size"],
            individual=False,
            d_model=dlinear_config["d_model"],
            ORT_weight=dlinear_config["ORT_weight"],
            MIT_weight=dlinear_config["MIT_weight"],
            batch_size=dlinear_config["batch_size"],
            epochs=dlinear_config["epochs"],
            patience=dlinear_config["patience"],
            # training_loss=<class 'pypots.nn.modules.loss.MAE'>, # unclear what authors used as they had an older version of PyPOTS
            # validation_metric=<class 'pypots.nn.modules.loss.MSE'>, # unclear what authors used as they had an older version of PyPOTS
            optimizer=Adam(**dlinear_config["optimizer_parameters"]),
            num_workers=0,
            device=DEVICE,
            saving_path=None,
            model_saving_strategy='best',
            verbose=True
        )
        model.fit(train_set)
        imputation = model.predict(val_set)["imputation"][..., 0]
        rows = to_imputed_hourly(val_set, val_index, imputation)

        daily_total_result = calculate_imputation_percent_errors(
            to_imputed_hourly_df=rows,
            indicators_included=False,
            daily_pool=True,
            window=(6, 22),
            include_outside_window=True,
        )
        daily_total_result.update(experiment_dict, run_dict)
        result_dict_list.append(daily_total_result)

        daily_open_result = calculate_imputation_percent_errors(
            to_imputed_hourly_df=rows,
            indicators_included=False,
            daily_pool=True,
            window=(6, 22),
            include_outside_window=False,
        )
        daily_open_result.update(experiment_dict, run_dict)
        result_dict_list.append(daily_open_result)

        hourly_total_result = calculate_imputation_percent_errors(
            to_imputed_hourly_df=rows,
            indicators_included=False,
            daily_pool=False,
            window=(6, 22),
            include_outside_window=True,
        )
        hourly_total_result.update(experiment_dict, run_dict)
        result_dict_list.append(hourly_total_result)

        hourly_open_result = calculate_imputation_percent_errors(
            to_imputed_hourly_df=rows,
            indicators_included=False,
            daily_pool=False,
            window=(6, 22),
            include_outside_window=False,
        )
        hourly_open_result.update(experiment_dict, run_dict)
        result_dict_list.append(hourly_open_result)

experimental_results = pd.DataFrame(result_dict_list)
experimental_results.to_csv(path_or_buf=(EXPERIMENT_DATA_DIRECTORY / "initial_dlinear_baseline_experiment.csv"))


# experimental_file_path = (
#     EXPERIMENT_DATA_DIRECTORY / "dlinear_baseline_evaluation_1_fold_1.csv"
# )
# rows = pd.read_csv(experimental_file_path, index_col=0)

# vanilla_results = calculate_imputation_percent_errors(
#     to_imputed_hourly_df=rows,
#     indicators_included=False,
#     daily_pool=True,
#     window=(6, 22),
#     include_outside_window=True,
# )
# stringent_daily_results = calculate_imputation_percent_errors(
#     to_imputed_hourly_df=rows,
#     indicators_included=False,
#     daily_pool=True,
#     window=(6, 22),
#     include_outside_window=False,
# )
# hourly_results = calculate_imputation_percent_errors(
#     to_imputed_hourly_df=rows,
#     indicators_included=False,
#     daily_pool=False,
#     window=(6, 22),
#     include_outside_window=True,
# )
# stringent_hourly_results = calculate_imputation_percent_errors(
#     to_imputed_hourly_df=rows,
#     indicators_included=False,
#     daily_pool=False,
#     window=(6, 22),
#     include_outside_window=False,
# )

# print(f"Vanilla Daily: {vanilla_results}")
# print(f"Stringent Daily: {stringent_daily_results}")
# print(f"Vanilla Hourly: {hourly_results}")
# print(f"Stringent Hourly: {stringent_hourly_results}")
