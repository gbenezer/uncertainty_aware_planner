# file to define the TimesNet Quantile model

from pathlib import Path
from typing import Dict, Sequence, Optional, Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from pypots.imputation.timesnet import TimesNet
from pypots.optim.adamw import AdamW

from uncertainty_aware_planner.models.imputation.pypots_quantile_loss import QuantileLoss


class TimesNet_Quantile:
    def __init__(self, cfg: Dict, saving_path: Optional[Union[Path, str]] = None):

        model_name = cfg["name"]
    
        if model_name != "TimesNet_Quantile":
            raise ValueError(f"Wrong Configuration: expected TimesNet_Quantile, got {model_name}")

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
            self.models[q] = TimesNet(
                n_steps=cfg["n_steps"], 
                n_features=cfg["n_features"], 
                n_layers=cfg["n_layers"], 
                top_k=cfg["top_k"], 
                d_model=cfg["d_model"], 
                d_ffn=cfg["d_ffn"], 
                n_kernels=cfg["n_kernels"], 
                dropout=cfg["dropout"], 
                apply_nonstationary_norm=cfg["apply_nonstationary_norm"], 
                batch_size=cfg["batch_size"], 
                epochs=cfg["epochs"], 
                patience=cfg["patience"], 
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