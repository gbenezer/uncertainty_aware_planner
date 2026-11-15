# file to define the TimesNet Quantile model

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from pypots.imputation.timesnet import TimesNet
from pypots.optim.adamw import AdamW

from uncertainty_aware_planner.models.imputation.pypots_quantile_loss import (
    QuantileLoss,
)

_REQUIRED_KEYS = {
    "name",
    "n_steps",
    "n_features",
    "n_layers",
    "top_k",
    "d_model",
    "d_ffn",
    "n_kernels",
    "dropout",
    "apply_nonstationary_norm",
    "batch_size",
    "epochs",
    "quantiles",
}


class TimesNet_Imputation_Quantile:
    def __init__(self, cfg: Dict, saving_path: Optional[Union[Path, str]] = None):
        missing = _REQUIRED_KEYS - cfg.keys()
        if missing:
            raise KeyError(f"Missing required config keys: {sorted(missing)}")

        model_name = cfg["name"]
        if model_name != "TimesNet_Imputation_Quantile":
            raise ValueError(
                f"Wrong Configuration: expected TimesNet_Imputation_Quantile, got {model_name}"
            )

        # TimesNet picks the top_k frequencies from an rfft over n_steps,
        # which only has n_steps // 2 + 1 bins
        if cfg["top_k"] > cfg["n_steps"] // 2 + 1:
            raise ValueError(
                f"top_k ({cfg['top_k']}) cannot exceed n_steps // 2 + 1 "
                f"({cfg['n_steps'] // 2 + 1})"
            )

        # Quantile validation: non-empty, strictly inside (0, 1), no duplicates,
        # and no filename-tag collisions at 1% resolution
        quantiles = sorted(set(float(q) for q in cfg["quantiles"]))
        if not quantiles:
            raise ValueError("cfg['quantiles'] must contain at least one quantile")
        if any(not (0.0 < q < 1.0) for q in quantiles):
            raise ValueError(
                f"Quantiles must be strictly between 0 and 1, got {quantiles}"
            )
        tags = [self._tag(q) for q in quantiles]
        if len(set(tags)) != len(tags):
            raise ValueError(
                f"Quantiles {quantiles} collide at 1% filename resolution: {tags}"
            )
        self.quantiles = tuple(quantiles)

        # Optional keys: a missing key and an explicit None behave the same
        optimizer_params = cfg.get("optimizer_parameters")
        patience = cfg.get("patience")

        self.saving_path = Path(saving_path) if saving_path is not None else None
        self.models = {}

        for q in self.quantiles:
            model_path = (
                str(self.saving_path / self._tag(q))
                if self.saving_path is not None
                else None
            )
            # fresh optimizer instance per model (never shared)
            optimizer = AdamW(**optimizer_params) if optimizer_params else AdamW()
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
                patience=patience,
                training_loss=QuantileLoss(quantile=q),
                validation_metric=QuantileLoss(quantile=q),
                optimizer=optimizer,
                num_workers=0,
                saving_path=model_path,
                model_saving_strategy="best",
                verbose=True,
            )

    @staticmethod
    def _tag(q: float) -> str:
        # consistent filenames: 0.1 -> q010, 0.5 -> q050, 0.9 -> q090
        return f"q{int(round(q * 100)):03d}"

    def _select(self, quantile: Optional[float] = None):
        # returns a list of (q, model) pairs: all of them, or just the requested one
        if quantile is None:
            return list(self.models.items())
        for q, m in self.models.items():
            if np.isclose(q, quantile):
                return [(q, m)]
        raise ValueError(
            f"Quantile {quantile} is not one of the configured quantiles "
            f"{list(self.quantiles)}"
        )

    def fit(
        self,
        train_set: Union[Dict, str],
        val_set: Optional[Union[Dict, str]] = None,
        quantile: Optional[float] = None,
    ):
        # quantile=None fits every sub-model; otherwise only the matching one
        selected = self._select(quantile)  # validate before any work is done
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
        for _, m in selected:
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

    def save(
        self,
        directory: Union[Path, str],
        overwrite: bool = False,
        quantile: Optional[float] = None,
    ):
        # Note: separate from PyPOTS's own timestamped checkpoints written
        # under saving_path/<tag> during fit().
        selected = self._select(quantile)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for q, m in selected:
            m.save(str(directory / f"{self._tag(q)}.pypots"), overwrite=overwrite)
        return self

    def load(self, directory: Union[Path, str]):
        directory = Path(directory)
        for q, m in self.models.items():
            m.load(str(directory / f"{self._tag(q)}.pypots"))
        return self
