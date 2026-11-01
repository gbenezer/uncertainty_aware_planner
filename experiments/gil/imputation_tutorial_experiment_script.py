import warnings
from pathlib import Path
import torch
import pandas as pd
from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam
from uncertainty_aware_planner.data.data_utilities import (
    load_raw_data_for_imputation,
    split_by_window_imputation,
)
from uncertainty_aware_planner.evaluation.imputation_performance import evaluation_mnar
from uncertainty_aware_planner.utils.constants import IMPUTATION_SPLITS
from uncertainty_aware_planner.utils.util_functions import set_seed


ROOT = Path("/home/benezer.gi/ondemand/dev/uncertainty_aware_planner")
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "experiments" / "gil" / "models"
OUT_DIR = ROOT / "experiments" / "gil" / "data"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# script-level constants
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CONFIG = {
    "saving_path": MODEL_DIR,
    "EPOCHS": 5,
    "batch_size": 128,
    "patience": 5,
    "d_model": 64,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "missing_rate": 0.3,
    "alpha": 1e-2,
}
SEEDS = [42, 235, 1024, 4, 16, 23]

warnings.simplefilter("ignore")
print(DEVICE)

# data loading
data = load_raw_data_for_imputation(
    missing_rate=CONFIG["missing_rate"], cache_dir="./data"
)
windowed_data_view = split_by_window_imputation(data)

for key, value in data["params"].items():
    print(key)
    print(value)

print("moving_avg_window_size")
output = CONFIG["patch_len"] // 2 * 2 + 1
print(output)

# updating the config
CONFIG.update(data["params"])

# creating a list for dataframe construction
dataframe_row_dictionaries = []

rows = []
for seed in SEEDS:
    for train_idx, eval_idx in IMPUTATION_SPLITS:
        set_seed(seed)

        # model construction
        model = DLinear(
            n_steps=CONFIG["n_steps"],
            n_features=CONFIG["n_features"],
            moving_avg_window_size=CONFIG["patch_len"] // 2 * 2 + 1,
            individual=False,
            d_model=CONFIG["d_model"],
            ORT_weight=1,
            MIT_weight=1,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["EPOCHS"],
            optimizer=Adam(lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]),
            device=DEVICE,
            saving_path=CONFIG["saving_path"],
            patience=CONFIG["patience"],
        )
        print(model.model)

        model.fit(train_set=windowed_data_view[train_idx]["train_set"])

        for partition, idx in (("train", train_idx), ("test", eval_idx)):
            preds = model.predict(test_set=windowed_data_view[idx]["train_set"])
            res = evaluation_mnar(
                input_data=windowed_data_view[idx],
                imputation_data=preds["imputation"],
                verbose=True,
            )
            res.update(
                train_window_idx=train_idx,
                eval_window_idx=idx,
                seed=seed,
                partition=partition,
            )
            rows.append(res)

        # incremental save so a crash doesn't lose finished runs
        pd.DataFrame(rows).to_csv(OUT_DIR / "dlinear_experiment_run_2.csv", index=False)
