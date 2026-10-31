import warnings
import torch
from pypots.imputation.dlinear import DLinear
from pypots.optim.adamw import AdamW
from uncertainty_aware_planner.data.data_utilities import (
    load_raw_data_for_imputation,
    split_by_window_imputation,
)
from uncertainty_aware_planner.evaluation.imputation_performance import evaluation_mnar
from uncertainty_aware_planner.utils.util_functions import set_seed

# script-level constants
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CONFIG = {
    "saving_path": "./models/",
    "EPOCHS": 10,
    "batch_size": 128,
    "patience": 5,
    "d_model": 64,
    "lr": 0.001,
    "missing_rate": 0.3,
    "alpha": 1e-2,
}
set_seed(42)

warnings.simplefilter("ignore")
print(DEVICE)

# data loading
data = load_raw_data_for_imputation(missing_rate=CONFIG["missing_rate"])
windowed_data_view = split_by_window_imputation(data)

# updating the config
CONFIG.update(data["params"])

# model definition
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
    optimizer=AdamW(lr=CONFIG["lr"]),
    device=DEVICE,
    saving_path=CONFIG["saving_path"],
    patience=CONFIG["patience"],
)

# training a model on a particular window
model.fit(train_set=windowed_data_view[2]["train_set"])

# imputing data in the same window
results_train_set = model.predict(test_set=windowed_data_view[0]["train_set"])

# checking the results on the evaluation of training data
eval_train_results = evaluation_mnar(
    input_data=windowed_data_view[0],
    imputation_data=results_train_set["imputation"],
    verbose=True,
)

# imputing and evaluating on a non-training window set
results_test_set = model.predict(test_set=windowed_data_view[1]["train_set"])
eval_test_results = evaluation_mnar(
    input_data=windowed_data_view[1],
    imputation_data=results_test_set["imputation"],
    verbose=True,
)
