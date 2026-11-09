# file to define the DLinear Quantile model

from pathlib import Path
from typing import Dict, Sequence, Optional, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam

from uncertainty_aware_planner.models.imputation.pypots_quantile_loss import QuantileLoss


class DLinear_Quantile:
    def __init__(self, cfg: Dict, saving_path: Optional[Union[Path, str]] = None):

        model_name = cfg["name"]
    
        if model_name != "DLinear_Quantile":
            raise ValueError(f"Wrong Configuration: expected DLinear_Quantile, got {model_name}")

        self.quantiles = tuple(sorted(cfg['quantiles']))
        self.models = {}

        if saving_path is not None:
            self.saving_path = saving_path if isinstance(saving_path, Path) else Path(saving_path)
        else:
            self.saving_path = None

        model_paths = {}
        
        for q in self.quantiles:
            if self.saving_path is not None:
                model_paths[q] = self.saving_path / f"q{q}"
            self.models[q] = DLinear(
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
                    training_loss=QuantileLoss(quantile=q),
                    validation_metric=QuantileLoss(quantile=q),
                    optimizer=AdamW(**cfg["optimizer_parameters"]),
                    saving_path=model_paths[q],
                    model_saving_strategy='best',
                    verbose=True
                )

    def fit(self, train_set, val_set: Optional[Union[Path, str]] = None):
        for q, m in self.models.items():
            m.fit(train_set, val_set)
        return self

    def impute(self, test_set, sort=True):
        # returns [n_samples, n_steps, n_features, n_quantiles]
        preds = np.stack(
            [m.impute(test_set) for m in self.models.values()], axis=-1
        )
        if sort:  # enforce non-crossing quantiles
            preds = np.sort(preds, axis=-1)
        return preds

    def save(self, directory: Union[Path, str], overwrite=False):
        
        if isinstance(directory, str):
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)

        for q, m in self.models.items():
            m.save((directory / f"q{q}.pypots"), overwrite=overwrite)

    def load(self, directory):
        for q, m in self.models.items():
            m.load((directory / f"q{q}.pypots"))