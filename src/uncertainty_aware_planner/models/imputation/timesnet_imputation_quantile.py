# file to define the TimesNet Quantile model

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from pypots.imputation.timesnet import TimesNet
from pypots.optim.adamw import AdamW

from uncertainty_aware_planner.models.imputation.pypots_quantile_loss import (
    QuantileLoss,
)


class TimesNet_Imputation_Quantile:
    def __init__(self, cfg: Dict, saving_path: Optional[Union[Path, str]] = None):
        model_name = cfg["name"]
        if model_name != "TimesNet_Imputation_Quantile":
            raise ValueError(
                f"Wrong Configuration: expected TimesNet_Imputation_Quantile, got {model_name}"
            )

        self.quantiles = tuple(sorted(cfg["quantiles"]))
        self.saving_path = Path(saving_path) if saving_path is not None else None
        self.models = {}

        for q in self.quantiles:
            model_path = (
                str(self.saving_path / self._tag(q))
                if self.saving_path is not None
                else None
            )
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
                optimizer=AdamW(
                    **cfg["optimizer_parameters"]
                ),  # fresh instance per model
                saving_path=model_path,
                model_saving_strategy="best",
                verbose=True,
            )

    @staticmethod
    def _tag(q: float) -> str:
        # consistent filenames: 0.1 -> q010, 0.5 -> q050, 0.9 -> q090
        return f"q{int(round(q * 100)):03d}"

    def fit(
        self, train_set: Union[Dict, str], val_set: Optional[Union[Dict, str]] = None
    ):
        for m in self.models.values():
            m.fit(train_set, val_set)
        return self

    def impute(self, test_set: Union[Dict, str], sort: bool = True) -> np.ndarray:
        # returns [n_samples, n_steps, n_features, n_quantiles]
        preds = np.stack([m.impute(test_set) for m in self.models.values()], axis=-1)
        if sort:  # enforce non-crossing quantiles
            preds = np.sort(preds, axis=-1)
        return preds

    def impute_point(self, test_set: Union[Dict, str]) -> np.ndarray:
        # median (q=0.5) is the point estimate
        # returns [n_samples, n_steps, n_features]
        for q, m in self.models.items():
            if np.isclose(q, 0.5):
                # only run the median model, skipping the other quantile models
                return m.impute(test_set)

        # no 0.5 quantile was trained: fall back to the median of the quantile predictions
        preds = self.impute(test_set)
        return np.median(preds, axis=-1)

    def save(self, directory: Union[Path, str], overwrite: bool = False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for q, m in self.models.items():
            m.save(str(directory / f"{self._tag(q)}.pypots"), overwrite=overwrite)

    def load(self, directory: Union[Path, str]):
        directory = Path(directory)
        for q, m in self.models.items():
            m.load(str(directory / f"{self._tag(q)}.pypots"))
