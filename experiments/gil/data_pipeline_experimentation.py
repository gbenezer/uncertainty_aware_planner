from pathlib import Path

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.helper_functions import to_imputed_hourly

# script constants
FOLD = 1

cfg = load_config()
cube = load_cube(cfg)
masks = pd.read_parquet("data/processed/eval_masks.parquet")

train_set, _ = build_pypots_set(cube, cfg, "train", fold=FOLD)  # {"X"}
val_set, val_index = build_pypots_set(cube, cfg, "val", fold=FOLD, masks=masks)



# e.g. pypots.imputation.DLinear(n_steps=672, n_features=9, ...); settings are yours
# model = ...
# model.fit(train_set)
# imputation = model.predict(val_set)["imputation"][..., 0]
# rows = to_imputed_hourly(val_set, val_index, imputation)