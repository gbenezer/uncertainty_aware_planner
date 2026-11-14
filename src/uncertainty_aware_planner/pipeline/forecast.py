"""Forecasting stage: train one model on one fold (or the final split) and write CONTRACTS.md 2.5 tables.

    python -m uncertainty_aware_planner.pipeline.forecast \
        --config configs/forecasting_models/temporal_fusion_transformer.yaml \
        --demand-daily data/processed/demand_daily/<imputation-run-id>.parquet \
        --seed 2025 [--fold 3 | --eval-parts calib test] [--smoke]

Fold k trains on the fold's train window and forecasts its val window (rolling 7-day origins). Without
--fold the final model trains on 03-28..06-18 and forecasts the calibration and test weeks from the same
checkpoint, so conformal offsets fitted on calib apply to the model that made the test forecasts.

Outputs: <output_dir>/<run_id>/{fold<k>,final_<part>}.parquet and <run_dir>/<run_id>/{fold<k>,final}/ with
the checkpoint, training logs, all predicted quantiles and run_log.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import time
from datetime import UTC, datetime

import pandas as pd
import torch

from uncertainty_aware_planner.data.config import load_config, load_yaml, resolve
from uncertainty_aware_planner.data.splits import final_windows
from uncertainty_aware_planner.data.timeseries_dataset import build_timeseries_datasets
from uncertainty_aware_planner.models.forecasting import tft, dlinear_quantile
from uncertainty_aware_planner.models.forecasting.forecast_table import (
    long_quantiles,
    run_id,
    to_forecast_table,
)
from uncertainty_aware_planner.utils.util_functions import set_seed

# Each architecture module provides build_model(train_ds, cfg), fit(model, train_ds, cfg, run_dir) -> ckpt
# and predict_quantiles(ckpt, eval_ds, cfg) -> (quantiles, index, decoder_lengths).
ARCHITECTURES = {
                "tft": tft,
                "dlinear_quantile": dlinear_quantile,}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def smoke_config(model_cfg: dict) -> dict:
    return _merge(model_cfg, model_cfg.get("smoke", {}))


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=resolve("."),
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _same_encoders(a, b) -> bool:
    enc_a, enc_b = (
        a.get_parameters()["categorical_encoders"],
        b.get_parameters()["categorical_encoders"],
    )
    return enc_a.keys() == enc_b.keys() and all(
        getattr(enc_a[k], "classes_", None) == getattr(enc_b[k], "classes_", None)
        for k in enc_a
    )


def run(
    model_cfg: dict,
    data_cfg: dict,
    demand_daily: pd.DataFrame,
    seed: int,
    fold: int | None = None,
    eval_parts: list[str] | None = None,
    demand_daily_path: str | None = None,
) -> dict:
    """Train, predict every eval part, write the tables; return the run log."""
    started = time.time()
    set_seed(seed)
    model_module = ARCHITECTURES[model_cfg["architecture"]]
    parts = ["val"] if fold is not None else list(eval_parts or ["calib", "test"])
    rid = run_id(model_cfg["model_id"], seed)
    run_dir = (
        resolve(model_cfg["run_dir"])
        / rid
        / (f"fold{fold}" if fold is not None else "final")
    )
    out_dir = resolve(model_cfg["output_dir"]) / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    start_date = final_windows(data_cfg["splits"])["train"].start

    train_ds, eval_ds, spec = build_timeseries_datasets(
        demand_daily, data_cfg, fold=fold, eval_part=parts[0]
    )
    model = model_module.build_model(train_ds, model_cfg)
    ckpt = model_module.fit(model, train_ds, model_cfg, run_dir)

    log = {
        "run_id": rid,
        "model_id": model_cfg["model_id"],
        "architecture": model_cfg["architecture"],
        "seed": seed,
        "fold": fold,
        "eval_parts": parts,
        "demand_daily": demand_daily_path,
        "imputation_run_id": str(demand_daily["imputation_run_id"].iloc[0]),
        "git_commit": _git_commit(),
        "model_config": model_cfg,
        "dataset_spec": spec,
        "n_train_samples": len(train_ds),
        "checkpoint": str(ckpt),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else platform.processor(),
        "outputs": {},
    }
    for i, part in enumerate(parts):
        if i > 0:
            part_train_ds, eval_ds, _ = build_timeseries_datasets(
                demand_daily, data_cfg, eval_part=part
            )
            if not _same_encoders(train_ds, part_train_ds):
                raise RuntimeError(
                    f"the {part} dataset encodes categories differently from training"
                )
        q, index, lengths = model_module.predict_quantiles(ckpt, eval_ds, model_cfg)
        rows, stats = long_quantiles(
            q, index, lengths, model_cfg["quantiles"], start_date
        )
        table = to_forecast_table(
            rows, demand_daily, model_cfg["model_id"], seed, fold, part
        )

        name = f"fold{fold}" if fold is not None else f"final_{part}"
        table_path = out_dir / f"{name}.parquet"
        table.to_parquet(table_path, index=False)
        rows.to_parquet(run_dir / f"quantiles_{part}.parquet", index=False)
        log["outputs"][part] = {
            "table": str(table_path),
            "n_eval_samples": len(eval_ds),
            **stats,
        }

    log["finished_at"] = datetime.now(UTC).isoformat()
    log["wall_seconds"] = round(time.time() - started, 1)
    with open(run_dir / "run_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    return log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        help="model config, e.g. configs/forecasting_models/...yaml",
    )
    parser.add_argument(
        "--demand-daily", required=True, help="demand_daily parquet (CONTRACTS.md 2.4)"
    )
    parser.add_argument("--seed", type=int, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fold", type=int, help="rolling fold 1..6; omit for the final split"
    )
    group.add_argument(
        "--eval-parts",
        nargs="+",
        choices=["calib", "test"],
        default=["calib", "test"],
    )
    parser.add_argument(
        "--data-config", help="defaults to data_config in the model config"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny model, few batches: checks the code path",
    )
    args = parser.parse_args(argv)

    model_cfg = load_yaml(args.config)
    if args.smoke:
        model_cfg = smoke_config(model_cfg)
    data_cfg = load_config(args.data_config or model_cfg["data_config"])
    demand = pd.read_parquet(resolve(args.demand_daily))
    log = run(
        model_cfg,
        data_cfg,
        demand,
        args.seed,
        fold=args.fold,
        eval_parts=None if args.fold is not None else args.eval_parts,
        demand_daily_path=args.demand_daily,
    )
    print(
        json.dumps(
            {
                "run_id": log["run_id"],
                "wall_seconds": log["wall_seconds"],
                **log["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
