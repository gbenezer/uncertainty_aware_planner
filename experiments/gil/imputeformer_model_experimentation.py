from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
import datetime

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.helper_functions import to_imputed_hourly
from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_percent_errors,
)
from uncertainty_aware_planner.utils.util_functions import set_seed, get_git_state
from uncertainty_aware_planner.models.imputation import (
    ImputeFormer_Imputation_Quantile,
)

# script constants

# model configs
IMPUTEFORMER_QUANTILE_CONFIG_FILE = Path(
    "configs/imputation_models/imputeformer_imputation_quantile.yaml"
)
IMPUTEFORMER_QUANTILE_MEDIAN_CONFIG_FILE = Path(
    "configs/imputation_models/imputeformer_imputation_quantile_median.yaml"
)

# data configs
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")

# output dirs
EXPERIMENT_DATA_DIRECTORY = Path("experiments/gil/data")
RESULTS_FILE = EXPERIMENT_DATA_DIRECTORY / "imputeformer_comparison_experiment_1.csv"
REPO_DIR = Path(__file__).resolve().parent

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEEDS = [86400, 12, 214, 8, 73, 4, 42, 2048, 1]

# Get a working imputation baseline experiment log commit file hash
SHA = get_git_state(REPO_DIR)

# resume: figure out which runs are already done
if RESULTS_FILE.exists():
    done = pd.read_csv(RESULTS_FILE, usecols=["seed", "fold"]).drop_duplicates()
    completed = set(zip(done["seed"], done["fold"]))
else:
    completed = set()


# loading config files for models
imputeformer_quantile_config = load_yaml(IMPUTEFORMER_QUANTILE_CONFIG_FILE)
imputeformer_quantile_median_config = load_yaml(IMPUTEFORMER_QUANTILE_MEDIAN_CONFIG_FILE)

# load necessary data pipeline elements
cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")

# constructing the experiment-wide metadata dictionary
experiment_dict = {}
experiment_dict["commit"] = SHA
experiment_dict["public_data_config_file"] = PUBLIC_DATA_CONFIG_FILE
experiment_dict["split_config_file"] = SPLIT_CONFIG_FILE
experiment_dict["device"] = DEVICE

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

        # retrieve the correct imputation train and validation data
        train_set, _ = build_pypots_set(cube, cfg, "train", fold=fold)
        val_set, val_index = build_pypots_set(cube, cfg, "val", fold=fold, masks=masks)

        # initialize the run results list
        run_results = []

        # evaluate on all models
        for imputeformer_model_config in [
            imputeformer_quantile_config,
            imputeformer_quantile_median_config
        ]:
            # set the seed
            set_seed(seed)

            # construct the model
            m = ImputeFormer_Imputation_Quantile(imputeformer_model_config)
            num_quantiles = len(imputeformer_model_config["quantiles"])


            # fit on the training dataset
            m.fit(train_set)

            # predict on the validation set
            out = m.impute_point(val_set)

            # evaluate on the validation set
            result = calculate_imputation_percent_errors(
                eval_pypots=val_set, imputation_point_estimate=out, daily_pool=False
            )

            # label whether or not the result is from leaked data or not
            result["num_quantiles"] = num_quantiles

            # update with metadata and add to the run results
            result.update({**experiment_dict, **run_dict})
            run_results.append(result)

        # one append per run, so a run is either fully recorded or not at all
        pd.DataFrame(run_results).to_csv(
            RESULTS_FILE,
            mode="a",
            header=not RESULTS_FILE.exists(),
            index=False,
        )
