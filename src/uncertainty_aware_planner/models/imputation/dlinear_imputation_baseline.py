# file to define the DLinear baseline model

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from pypots.imputation.dlinear import DLinear
from pypots.optim.adamw import AdamW

_REQUIRED_KEYS = {
    "name",
    "n_steps",
    "n_features",
    "moving_avg_window_size",
    "individual",
    "d_model",
    "ORT_weight",
    "MIT_weight",
    "batch_size",
    "epochs",
}


class DLinear_Imputation_Baseline:
    def __init__(self, cfg: Dict, saving_path: Optional[Union[Path, str]] = None):
        missing = _REQUIRED_KEYS - cfg.keys()
        if missing:
            raise KeyError(f"Missing required config keys: {sorted(missing)}")

        model_name = cfg["name"]
        if model_name != "DLinear_Imputation_Baseline":
            raise ValueError(
                f"Wrong Configuration: expected DLinear_Imputation_Baseline, got {model_name}"
            )

        if cfg["moving_avg_window_size"] % 2 == 0:
            raise ValueError("moving_avg_window_size must be odd")

        # Optional keys: a missing key and an explicit None behave the same
        optimizer_params = cfg.get("optimizer_parameters")
        optimizer = AdamW(**optimizer_params) if optimizer_params else AdamW()
        patience = cfg.get("patience")

        self.saving_path = Path(saving_path) if saving_path is not None else None

        self.model = DLinear(
            n_steps=cfg["n_steps"],
            n_features=cfg["n_features"],
            moving_avg_window_size=cfg["moving_avg_window_size"],
            individual=cfg["individual"],
            d_model=cfg["d_model"],
            ORT_weight=cfg["ORT_weight"],
            MIT_weight=cfg["MIT_weight"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            patience=patience,
            # training_loss / validation_metric left at PyPOTS defaults (MAE / MSE)
            optimizer=optimizer,
            num_workers=0,
            saving_path=str(self.saving_path) if self.saving_path is not None else None,
            model_saving_strategy="best",
            verbose=True,
        )

    def fit(
        self, train_set: Union[Dict, str], val_set: Optional[Union[Dict, str]] = None
    ):
        if val_set is None:
            warnings.warn(
                "No val_set given: early stopping and best-model saving will "
                "fall back to training loss.",
                stacklevel=2,
            )
        elif isinstance(val_set, dict) and "X_ori" not in val_set:
            raise ValueError(
                "val_set for imputation must contain 'X' (artificially masked) "
                "and 'X_ori' (ground truth)."
            )
        self.model.fit(train_set, val_set)
        return self

    def impute(self, test_set: Union[Dict, str]) -> np.ndarray:
        # returns [n_samples, n_steps, n_features]
        return self.model.impute(test_set)

    def impute_point(self, test_set: Union[Dict, str]) -> np.ndarray:
        # alias for impute
        return self.impute(test_set)

    def save(self, directory: Union[Path, str], overwrite: bool = False):
        # Note: this is separate from PyPOTS's own timestamped checkpoints
        # written under saving_path during fit().
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(
            str(directory / "dlinear_imputation_baseline.pypots"), overwrite=overwrite
        )
        return self

    def load(self, directory: Union[Path, str]):
        directory = Path(directory)
        self.model.load(str(directory / "dlinear_imputation_baseline.pypots"))
        return self
