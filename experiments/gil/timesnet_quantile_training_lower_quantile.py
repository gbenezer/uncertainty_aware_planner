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
    TimesNet_Imputation_Quantile,
)

# script constants

# model configs
TIMESNET_QUANTILE_MEDIAN_CONFIG_FILE = Path(
    "configs/imputation_models/timesnet_imputation_quantile_median.yaml"
)

# data configs
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")

# output dirs
MODEL_DIRECTORY = Path("experiments/gil/models")
REPO_DIR = Path(__file__).resolve().parent

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEEDS = [86400, 4, 42]

# Get a working imputation baseline experiment log commit file hash
SHA = get_git_state(REPO_DIR)

# loading config files for model
timesnet_quantile_median_config = load_yaml(TIMESNET_QUANTILE_MEDIAN_CONFIG_FILE)
training_quantile = timesnet_quantile_median_config["quantiles"][0]

# load necessary data pipeline elements
cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")

for seed in SEEDS:
    # retrieve the correct imputation train and validation data
    train_set, _ = build_pypots_set(cube, cfg, "train")

    # set the seed
    set_seed(seed)

    # construct the model
    m = TimesNet_Imputation_Quantile(timesnet_quantile_median_config)

    # fit on the training dataset
    m.fit(train_set, quantile=training_quantile)

    RUN_DIRECTORY = MODEL_DIRECTORY / f"seed_{seed}_timesnet"
    m.save(directory=RUN_DIRECTORY, quantile=training_quantile)
