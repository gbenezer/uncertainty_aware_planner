"""constructed independently by repurposing Claude-generated
code from pretrained_model_eval.py; I ran out of credits until 30Nov
so had to make stuff work manually"""

import datetime
from pathlib import Path

import pandas as pd

from uncertainty_aware_planner.data.config import load_config, load_yaml
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.utils.util_functions import set_seed
from uncertainty_aware_planner.evaluation import imputation_evaluation_helpers as ie

# ---------------------------------------------------------------- constants
CONFIG_DIR = Path("configs/imputation_models")
CONFIG_FILES = [
    "dlinear_imputation_quantile.yaml",
    "dlinear_imputation_quantile_median.yaml",
    "timesnet_imputation_quantile_median.yaml",
    "dlinear_imputation_baseline.yaml",
]
PUBLIC_DATA_CONFIG_FILE = Path("configs/data.yaml")
SPLIT_CONFIG_FILE = Path("configs/splits.yaml")

DATA_DIR = Path("data/hf_cache")
MODEL_DIRECTORY = Path("experiments/gil/models")
OUTPUT_DIR = Path("results/imputation/pretrained_model_eval/model_outputs")
OUTPUT_STEM = "pretrained_model_inference_output"
REPO_DIR = Path(__file__).resolve().parent

SEEDS = [86400, 4, 42]

# (model name, num quantiles) -> subdirectory suffix of the pretrained checkpoint
CHECKPOINT_SUBDIRS = {
    ("DLinear_Imputation_Baseline", None): "dlinear_baseline",
    ("DLinear_Imputation_Quantile", 2): "dlinear_quantile_interval_only",
    ("DLinear_Imputation_Quantile", 3): "dlinear_quantile",
    ("TimesNet_Imputation_Quantile", 3): "timesnet",
}


# ---------------------------------------------------------------------- processing code
configs = [load_yaml(CONFIG_DIR / f) for f in CONFIG_FILES]
cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")
calib_set, calib_index = build_pypots_set(cube, cfg, "calib", masks=masks)

for seed in SEEDS:
    for config in configs:
        model_type = config["name"]
        quantiles = config.get("quantiles")
        num_quantiles = len(quantiles) if quantiles is not None else None

        # not sure this does anything post-training, but it can't hurt
        set_seed(seed)

        # load model
        key = (model_type, num_quantiles)
        model = ie.MODEL_CLASSES[model_type](config)
        model.load(MODEL_DIRECTORY / f"seed_{seed}_{CHECKPOINT_SUBDIRS[key]}")

        # impute data
        imputed_data = model.impute(test_set=calib_set)

        # convert it to an appropriate hourly dataframe
        imputed_df = ie.TO_DATAFRAME[num_quantiles](
            calib_set, calib_index, imputed_data
        )

        # save the dataframe in the appropriate directory
        file_name = (
            OUTPUT_STEM + f"_seed_{seed}_{model_type}_{num_quantiles}_quantiles.csv"
        )
        file_path = OUTPUT_DIR / file_name
        imputed_df.to_csv(file_path)
