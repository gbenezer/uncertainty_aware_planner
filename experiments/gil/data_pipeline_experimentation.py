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
DLINEAR_CONFIG_FILE = Path("configs/imputation_models/dlinear_imputation_baseline.yaml")
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")
EXPERIMENT_DATA_DIRECTORY = Path("experiments/gil/data")
RESULTS_FILE = EXPERIMENT_DATA_DIRECTORY / "second_dlinear_baseline_experiment.csv"
REPO_DIR = Path(__file__).resolve().parent

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEEDS = [42, 2048, 1]


# temporary helper function to get to a working imputation baseline experiment log
# commit file
def get_git_state(repo_dir: Path = REPO_DIR) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return sha + ("-dirty" if status else "")

SHA = get_git_state()

# resume: figure out which runs are already done
if RESULTS_FILE.exists():
    done = pd.read_csv(RESULTS_FILE, usecols=["seed", "fold"]).drop_duplicates()
    completed = set(zip(done["seed"], done["fold"]))
else:
    completed = set()


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
experiment_dict.update({f"hyperparameter_{k}": v for k, v in dlinear_config.items()})

# initialize an empty list to hold results
result_dict_list = []

for seed in SEEDS:
    for fold in range(1, 7):

        if (seed, fold) in completed:
            print(f"Skipping seed={seed}, fold={fold} (already done)")
            continue

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

        # fit the model and impute
        model.fit(train_set)
        imputation = model.predict(val_set)["imputation"][..., 0]

        # construct the proper dataframe for evaluation
        rows = to_imputed_hourly(val_set, val_index, imputation)

        # calculate errors with both daily pooling and hourly,
        # as well as for both considering all hours of the day and only
        # hours where the store is open
        run_results = []
        for daily_pool, include_outside in [(True, True), (True, False), (False, True), (False, False)]:
            result = calculate_imputation_percent_errors(
                to_imputed_hourly_df=rows,
                indicators_included=False,
                daily_pool=daily_pool,
                window=(6, 22),
                include_outside_window=include_outside,
            )
            result.update({**experiment_dict, **run_dict})
            run_results.append(result)

        # one append per run, so a run is either fully recorded or not at all
        pd.DataFrame(run_results).to_csv(
            RESULTS_FILE,
            mode="a",
            header=not RESULTS_FILE.exists(),
            index=False,
        )

# experimental_results = pd.DataFrame(result_dict_list)
# experimental_results.to_csv(path_or_buf=(EXPERIMENT_DATA_DIRECTORY / "initial_dlinear_baseline_experiment.csv"))


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
