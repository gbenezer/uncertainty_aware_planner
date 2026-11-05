from pathlib import Path

import numpy as np
import pandas as pd
import torch
import datetime

# from pypots.imputation.dlinear import DLinear
# from pypots.optim.adam import Adam

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.helper_functions import to_imputed_hourly
from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_percent_errors,
)

# script constants
FOLD = 1
DLINEAR_CONFIG_FILE = Path("configs/imputation_models/dlinear_imputation_baseline.yaml")
EXPERIMENT_DATA_DIRECTORY = Path("experiments/gil/data")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# # loading config file
# dlinear_config = load_yaml(DLINEAR_CONFIG_FILE)

# # load necessary data pipeline elements
# cfg = load_config()
# cube = load_cube(cfg)
# masks = pd.read_parquet("data/processed/eval_masks.parquet")

# # test
# train_set, _ = build_pypots_set(cube, cfg, "train", fold=FOLD)
# val_set, val_index = build_pypots_set(cube, cfg, "val", fold=FOLD, masks=masks)

# model = DLinear(
#     n_steps = dlinear_config["n_steps"],
#     n_features = dlinear_config["n_features"],
#     moving_avg_window_size = dlinear_config["moving_avg_window_size"],
#     individual=False,
#     d_model=dlinear_config["d_model"],
#     ORT_weight=dlinear_config["ORT_weight"],
#     MIT_weight=dlinear_config["MIT_weight"],
#     batch_size=dlinear_config["batch_size"],
#     epochs=dlinear_config["epochs"],
#     patience=dlinear_config["patience"],
#     # training_loss=<class 'pypots.nn.modules.loss.MAE'>,
#     # validation_metric=<class 'pypots.nn.modules.loss.MSE'>,
#     optimizer=Adam(**dlinear_config["optimizer_parameters"]),
#     num_workers=0,
#     device=DEVICE,
#     saving_path=None,
#     model_saving_strategy='best',
#     verbose=True
# )
# model.fit(train_set)
# imputation = model.predict(val_set)["imputation"][..., 0]
# rows = to_imputed_hourly(val_set, val_index, imputation)

# for fold in range(1, 7):
#     train_set, _ = build_pypots_set(cube, cfg, "train", fold=fold)
#     val_set, val_index = build_pypots_set(cube, cfg, "val", fold=fold, masks=masks)
#     model = DLinear(
#         n_steps = dlinear_config["n_steps"],
#         n_features = dlinear_config["n_features"],
#         moving_avg_window_size = dlinear_config["moving_avg_window_size"],
#         individual=False,
#         d_model=dlinear_config["d_model"],
#         ORT_weight=dlinear_config["ORT_weight"],
#         MIT_weight=dlinear_config["MIT_weight"],
#         batch_size=dlinear_config["batch_size"],
#         epochs=dlinear_config["epochs"],
#         patience=dlinear_config["patience"],
#         # training_loss=<class 'pypots.nn.modules.loss.MAE'>,
#         # validation_metric=<class 'pypots.nn.modules.loss.MSE'>,
#         optimizer=Adam(**dlinear_config["optimizer_parameters"]),
#         num_workers=0,
#         device=DEVICE,
#         saving_path=None,
#         model_saving_strategy='best',
#         verbose=True
#     )
#     model.fit(train_set)
#     imputation = model.predict(val_set)["imputation"][..., 0]
#     rows = to_imputed_hourly(val_set, val_index, imputation)
#     experimental_file_path = EXPERIMENT_DATA_DIRECTORY / f"dlinear_baseline_evaluation_1_fold_{fold}.csv"
#     rows.to_csv(path_or_buf = experimental_file_path)

experimental_file_path = (
    EXPERIMENT_DATA_DIRECTORY / "dlinear_baseline_evaluation_1_fold_1.csv"
)
rows = pd.read_csv(experimental_file_path, index_col=0)

vanilla_results = calculate_imputation_percent_errors(
    to_imputed_hourly_df=rows,
    indicators_included=False,
    daily_pool=True,
    window=(6, 22),
    include_outside_window=True,
)
stringent_daily_results = calculate_imputation_percent_errors(
    to_imputed_hourly_df=rows,
    indicators_included=False,
    daily_pool=True,
    window=(6, 22),
    include_outside_window=False,
)
hourly_results = calculate_imputation_percent_errors(
    to_imputed_hourly_df=rows,
    indicators_included=False,
    daily_pool=False,
    window=(6, 22),
    include_outside_window=True,
)
stringent_hourly_results = calculate_imputation_percent_errors(
    to_imputed_hourly_df=rows,
    indicators_included=False,
    daily_pool=False,
    window=(6, 22),
    include_outside_window=False,
)

print(f"Vanilla Daily: {vanilla_results}")
print(f"Stringent Daily: {stringent_daily_results}")
print(f"Vanilla Hourly: {hourly_results}")
print(f"Stringent Hourly: {stringent_hourly_results}")
