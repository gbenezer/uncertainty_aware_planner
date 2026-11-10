# file to define the DLinear baseline model

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam


class DLinear_Imputation_Baseline:
    def __init__(self, cfg: Dict, saving_path: Optional[Union[Path, str]] = None):
        model_name = cfg["name"]
        if model_name != "DLinear_Imputation_Baseline":
            raise ValueError(
                f"Wrong Configuration: expected DLinear_Imputation_Baseline, got {model_name}"
            )

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
            patience=cfg["patience"],
            # training_loss / validation_metric left at PyPOTS defaults (MAE / MSE)
            optimizer=Adam(**cfg["optimizer_parameters"]),
            num_workers=0,
            saving_path=str(self.saving_path) if self.saving_path is not None else None,
            model_saving_strategy="best",
            verbose=True,
        )

    def fit(
        self, train_set: Union[Dict, str], val_set: Optional[Union[Dict, str]] = None
    ):
        self.model.fit(train_set, val_set)
        return self

    def impute(self, test_set: Union[Dict, str]) -> np.ndarray:
        # returns [n_samples, n_steps, n_features]
        return self.model.impute(test_set)

    def impute_point(self, test_set: Union[Dict, str]) -> np.ndarray:
        # alias for impute
        return self.impute(test_set)

    def save(self, directory: Union[Path, str], overwrite: bool = False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(
            str(directory / "dlinear_imputation_baseline.pypots"), overwrite=overwrite
        )

    def load(self, directory: Union[Path, str]):
        directory = Path(directory)
        self.model.load(str(directory / "dlinear_imputation_baseline.pypots"))
