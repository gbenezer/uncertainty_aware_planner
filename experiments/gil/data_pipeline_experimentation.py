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
    evaluation_mnar
)
from uncertainty_aware_planner.utils.util_functions import set_seed

# script constants
DLINEAR_CONFIG_FILE = Path("configs/imputation_models/dlinear_imputation_baseline.yaml")
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")
EXPERIMENT_DATA_DIRECTORY = Path("experiments/gil/data")
RESULTS_FILE = EXPERIMENT_DATA_DIRECTORY / "fourth_dlinear_baseline_experiment.csv"
REPO_DIR = Path(__file__).resolve().parent

# other variables
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEEDS = [86400, 12, 214, 8, 73, 4, 42, 2048, 1]


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

# # temporary code prototype derived from authors' repository copy-pasted
# def evaluation_mnar_old(X, imputation):

#     # download the full dataset
#     dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")

#     # change the training set to a Pandas dataframe in-memory
#     data = dataset['train'].to_pandas()

#     # sort by store-product-day
#     data = data.sort_values(by=['store_id', 'product_id', 'dt'])

#     # get the number of series in the data
#     series_num = data.shape[0]//HORIZON

#     # get the actual scaled sales every hour as well as the stockout status
#     hours_sale = np.array(data['hours_sale'].tolist())
#     hours_stock_status = np.array(data['hours_stock_status'].tolist())

#     # reshape the original hours into the same shape as expected by the input,
#     # which includes off hours
#     hours_sale_origin = hours_sale.reshape(series_num*WINDOW_NUMBER, IMPUTATION_WINDOW_DURATION, NUMBER_HOURS_IN_DAY)

#     # get the masking hours
#     stock_hour_X = (
#         np.isnan(X[...,0] # boolean to denote where sales in evaluation data are NaN
#         .reshape(-1,IMPUTATION_WINDOW_DURATION, NUMBER_OPEN_HOURS)) # reshape to something x days x open hours
#         .sum(axis=-1) # sum across hours
#         .reshape(-1, HORIZON) # reshape to something x Horizon (90 days)
#         .reshape(-1) # flatten
#         )

#     # variable where number of out of stock hours is summed across days during work hours
#     stock_hour_origin = hours_stock_status[:,6:22].sum(axis=1)

#     # copy the original unmasked data
#     hours_sale_impute = hours_sale_origin.copy()

#     # fill in the working hours with the imputation
#     hours_sale_impute[...,6:22] = imputation.reshape(-1, IMPUTATION_WINDOW_DURATION, NUMBER_OPEN_HOURS)

#     # sum the aggregate sales amounts across hours and reshape
#     sale_amount_pred = (
#         hours_sale_impute
#         .sum(axis=-1) # sum across the hours
#         .reshape(-1, HORIZON) # reshape to something x Horizon length (90 days)
#         .reshape(-1) # flatten
#         )
    
#     # valid indicies are where there are some masked hours in the day,
#     # but no true stockout
#     valid_idx = (stock_hour_X>0)&(stock_hour_origin==0)

#     # get the daily sales amount on a per day basis directly from the dataset
#     sale_amount = data['sale_amount'].values

#     return {
#         'wape': (np.abs(sale_amount_pred-sale_amount)*valid_idx).sum()/(sale_amount*valid_idx).sum(),
#         'wpe': ((sale_amount_pred-sale_amount)*valid_idx).sum()/(sale_amount*valid_idx).sum()
#     }

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
        model_output = model.predict(val_set)
        imputation = model_output["imputation"][..., 0]

        # construct the proper dataframe for evaluation
        rows = to_imputed_hourly(val_set, val_index, imputation)

        # calculate errors with both daily pooling and hourly,
        # as well as for both considering all hours of the day and only
        # hours where the store is open
        run_results = []
        for indicators_included, daily_pool, include_outside in [(False, True, True), (False, True, False), (False, False, True), (False, False, False), (True, True, True), (True, True, False), (True, False, True), (True, False, False)]:
            result = calculate_imputation_percent_errors(
                to_imputed_hourly_df=rows,
                indicators_included=indicators_included,
                daily_pool=daily_pool,
                window=(6, 22),
                include_outside_window=include_outside,
            )
            result["indicators_included"] = indicators_included
            result["daily_pool"] = daily_pool
            result["include_outside"] = include_outside
            result["raw_PyPOTS"] = False
            result.update({**experiment_dict, **run_dict})
            run_results.append(result)

        for daily_pool in [True, False]:
            result = evaluation_mnar(
                eval_pypots = val_set,
                model_output=model_output,
                daily_pool=daily_pool,
            )
            result["indicators_included"] = True
            result["daily_pool"] = daily_pool
            result["include_outside"] = False
            result["raw_PyPOTS"] = True
            result.update({**experiment_dict, **run_dict})
            run_results.append(result)

        # one append per run, so a run is either fully recorded or not at all
        pd.DataFrame(run_results).to_csv(
            RESULTS_FILE,
            mode="a",
            header=not RESULTS_FILE.exists(),
            index=False,
        )
