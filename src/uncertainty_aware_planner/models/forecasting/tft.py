"""Temporal Fusion Transformer (pytorch-forecasting) with a quantile loss.

Follows the dataset authors' demand_forecasting/TFT (trainTFT.py, configs/tft_config.py): fixed epochs and no
validation set during training. Here that also keeps the val/calib weeks out of model selection, since the
calibration week is reserved for conformal prediction.
"""

from __future__ import annotations

import pickle
import signal
import sys
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


def build_model(train_ds: TimeSeriesDataSet, cfg: dict) -> TemporalFusionTransformer:
    """output_size is deduced from the loss (one output per quantile)."""
    return TemporalFusionTransformer.from_dataset(
        train_ds, loss=QuantileLoss(quantiles=list(cfg["quantiles"])), **cfg["model"]
    )


def _trainer(cfg: dict, run_dir: Path, **overrides) -> pl.Trainer:
    kwargs = {k: v for k, v in cfg["trainer"].items() if v is not None}
    kwargs.setdefault("enable_progress_bar", sys.stdout.isatty())
    plugins = []
    # Launched with srun: save hpc_ckpt_*.ckpt and requeue on SIGUSR1 before the time limit.
    if SLURMEnvironment.detect():
        plugins.append(SLURMEnvironment(requeue_signal=signal.SIGUSR1))
    kwargs.update(overrides)
    return pl.Trainer(default_root_dir=str(run_dir), plugins=plugins, **kwargs)


def fit(
    model: TemporalFusionTransformer,
    train_ds: TimeSeriesDataSet,
    cfg: dict,
    run_dir: Path,
) -> Path:
    """Train for the configured epochs and return the final checkpoint.

    A requeued SLURM job resumes from the hpc_ckpt_*.ckpt Lightning wrote into run_dir.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Dataset parameters, to rebuild datasets for inference later (TimeSeriesDataSet.from_parameters).
    with open(run_dir / "dataset_params.pkl", "wb") as f:
        pickle.dump(train_ds.get_parameters(), f)

    loader = cfg["dataloader"]
    train_dl = train_ds.to_dataloader(
        train=True,
        batch_size=loader["batch_size"],
        num_workers=loader["num_workers"],
        persistent_workers=loader["num_workers"] > 0,
    )
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints", save_top_k=0, save_last=True
    )
    trainer = _trainer(
        cfg,
        run_dir,
        callbacks=[checkpoint],
        logger=CSVLogger(run_dir, name="logs"),
        limit_val_batches=0,
    )
    # pytorch-forecasting checkpoints pickle normalizers and encoders, which torch>=2.6 refuses to load
    # with weights_only=True (pytorch-forecasting #2463). Only our own checkpoints are loaded here.
    trainer.fit(model, train_dataloaders=train_dl, weights_only=False)
    return Path(checkpoint.last_model_path)


def predict_quantiles(
    ckpt_path: Path, eval_ds: TimeSeriesDataSet, cfg: dict
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Return (quantiles (n, horizon, n_quantiles), index, decoder lengths) for every sample in eval_ds.

    The model is reloaded for each call: predicting twice with one instance can return nothing
    (pytorch-forecasting #2108).
    """
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    loader = cfg["dataloader"]
    eval_dl = eval_ds.to_dataloader(
        train=False, batch_size=loader["batch_size"], num_workers=loader["num_workers"]
    )
    trainer_cfg = cfg["trainer"]
    prediction = model.predict(
        eval_dl,
        mode="quantiles",
        return_index=True,
        return_decoder_lengths=True,
        trainer_kwargs={
            "accelerator": trainer_cfg.get("accelerator") or "auto",
            "devices": 1,
            "logger": False,
        },
    )
    q = prediction.output.detach().cpu().numpy()
    lengths = prediction.decoder_lengths.detach().cpu().numpy()
    return q, prediction.index.reset_index(drop=True), lengths
