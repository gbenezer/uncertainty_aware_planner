"""Imputation stage (Feature 1): train one imputation model for one seed on one fold (or the final split) and
write CONTRACTS.md 2.3 `imputed_hourly` tables. The model-backed counterpart of pipeline/censored.py.

    python -m uncertainty_aware_planner.pipeline.impute \
        --config configs/imputation_models/dlinear_imputation_quantile.yaml --seed 86400 [--fold 3]

Fold k fits on the fold's train window and imputes its val window with the shared eval masks hidden (data
README 3.2). Without --fold, the final model fits on 03-28..06-18. From that same model it imputes the
calibration week with its masks hidden (3.3; conformal calibration reads it) and every real stockout hour
03-28..06-25 (3.4; `python -m uncertainty_aware_planner.data daily` turns that into demand_daily). The masked
hours' true sales are never passed to fit.

Outputs, never overwritten, each with a .manifest.json:
    <processed>/imputation/<run_id>/fold<k>.parquet      fold k val window: masked and stockout hours
    <processed>/imputation/<run_id>/final_calib.parquet  calibration week: masked and stockout hours
    <processed>/imputed_hourly/<run_id>.parquet          every real stockout hour, the `data daily` input
    data/runs/imputation/<run_id>/{fold<k>,final}/       model files and run_log.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pypots
import torch

from uncertainty_aware_planner.data.config import load_config, load_yaml, resolve
from uncertainty_aware_planner.data.cube import HIERARCHY, Cube, load_cube
from uncertainty_aware_planner.data.daily import recovered_demand
from uncertainty_aware_planner.data.helper_functions import _hidden_hours
from uncertainty_aware_planner.data.outputs import (
    file_sha256,
    write_manifest,
    write_parquet,
)
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.splits import (
    final_split_labels,
    final_windows,
    fold_windows,
)
from uncertainty_aware_planner.models.imputation import (
    DLinear_Imputation_Baseline,
    DLinear_Imputation_Quantile,
    TimesNet_Imputation_Quantile,
)
from uncertainty_aware_planner.utils.util_functions import get_git_state, set_seed

# Config `name` -> (model class, CONTRACTS.md model_id <feature>-<architecture>-<variant>).
ARCHITECTURES = {
    "DLinear_Imputation_Baseline": (
        DLinear_Imputation_Baseline,
        "imp-dlinear-baseline",
    ),
    "DLinear_Imputation_Quantile": (
        DLinear_Imputation_Quantile,
        "imp-dlinear-quantile",
    ),
    "TimesNet_Imputation_Quantile": (
        TimesNet_Imputation_Quantile,
        "imp-timesnet-quantile",
    ),
}
BASELINE = "DLinear_Imputation_Baseline"
COLUMNS = HIERARCHY + [
    "date",
    "time",
    "datetime",
    "impute_model_id",
    "impute_seed",
    "impute_fold",
    "impute_split",
    "q025",
    "q975",
    "estimated_demand",
    "actual_demand",
    "q025_cal",
    "q975_cal",
    "qhat95",
]
LOWER, UPPER = 0.025, 0.975
RUN_ROOT = "data/runs/imputation"
# Samples per model.impute call. A call returns every feature (and every quantile), so the full recovery
# set in one call would hold several copies of its 4.8 GB input.
PREDICT_CHUNK = 10_000


def _architecture(model_cfg: dict) -> tuple[type, str]:
    name = model_cfg.get("name")
    if name not in ARCHITECTURES:
        raise ValueError(
            f"unknown imputation model {name!r}; expected one of {sorted(ARCHITECTURES)}"
        )
    return ARCHITECTURES[name]


def model_id(model_cfg: dict) -> str:
    """The CONTRACTS.md model_id. Configs that also train the median get a -median suffix, so their run ids
    (and output files) differ from the two-quantile config of the same architecture."""
    _, base = _architecture(model_cfg)
    levels = model_cfg.get("quantiles") or []
    return f"{base}-median" if any(np.isclose(q, 0.5) for q in levels) else base


def quantile_levels(model_cfg: dict) -> list[float] | None:
    """Ascending quantile levels, as the model classes order their outputs; None for the baseline."""
    if model_cfg["name"] == BASELINE:
        return None
    levels = sorted({float(q) for q in model_cfg["quantiles"]})
    for level in (LOWER, UPPER):
        if not any(np.isclose(q, level) for q in levels):
            raise ValueError(
                f"the imputed_hourly table needs quantiles {LOWER} and {UPPER}; the config has {levels}"
            )
    return levels


def _position(levels: list[float], level: float) -> int:
    return next(k for k, q in enumerate(levels) if np.isclose(q, level))


def impute_sales(model, dataset: dict, chunk: int = PREDICT_CHUNK) -> np.ndarray:
    """Imputed sales (feature 0): (n_samples, n_steps) for the baseline, (n_samples, n_steps, n_quantiles) in
    ascending, non-crossing order for the quantile models."""
    x = dataset["X"]
    return np.concatenate(
        [
            model.impute({"X": x[i : i + chunk]})[:, :, 0]
            for i in range(0, len(x), chunk)
        ]
    )


def imputed_table(
    dataset: dict,
    index: pd.DataFrame,
    sales: np.ndarray,
    cube: Cube,
    levels: list[float] | None,
    *,
    model_id: str,
    seed: int,
    fold: int | None,
    split: str | None,
    splits_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """CONTRACTS.md 2.3 rows for every hidden hour a sample owns: real stockouts (actual_demand null) and
    masked in-stock hours (actual_demand = true sales). split None labels each row with its final split.

    Values are clipped at 0 (the clip rates go to the run log); estimated_demand is the midpoint of q025 and
    q975, or the baseline's own output.
    """
    base, sample, step = _hidden_hours(dataset, index)
    raw = sales[sample, step].astype(np.float64)
    n = len(base)
    if levels is None:
        lower = upper = np.full(n, np.nan)
        clipped = {"estimated_demand": raw}
    else:
        lower = raw[:, _position(levels, LOWER)]
        upper = raw[:, _position(levels, UPPER)]
        clipped = {"q025": lower, "q975": upper}
    clip_rate = {k: float((v < 0).mean()) if n else 0.0 for k, v in clipped.items()}
    lower, upper = np.clip(lower, 0.0, None), np.clip(upper, 0.0, None)
    # As forecast_table.py: the midpoint of the clipped interval, so q025 <= estimate <= q975 holds exactly.
    estimate = np.clip(raw, 0.0, None) if levels is None else (lower + upper) / 2

    positions = cube.series_index(base["store_id"], base["product_id"])
    table = cube.series[HIERARCHY].iloc[positions].reset_index(drop=True)
    table["date"] = base["date"]
    table["time"] = base["hour"]
    table["datetime"] = base["datetime"]
    table["impute_model_id"] = model_id
    table["impute_seed"] = pd.array([seed] * n, dtype="UInt32")
    table["impute_fold"] = pd.array([fold] * n, dtype="UInt8")
    table["impute_split"] = (
        split
        if split is not None
        else final_split_labels(splits_cfg, base["date"]).to_numpy(dtype=object)
    )
    table["q025"] = lower
    table["q975"] = upper
    table["estimated_demand"] = estimate
    table["actual_demand"] = base["actual_demand"].to_numpy(dtype=np.float64)
    for column in ("q025_cal", "q975_cal", "qhat95"):
        table[column] = np.nan
    table = (
        table[COLUMNS]
        .sort_values(["store_id", "product_id", "datetime"])
        .reset_index(drop=True)
    )

    if (
        levels is not None
        and not (
            (table["q025"] <= table["estimated_demand"])
            & (table["estimated_demand"] <= table["q975"])
        ).all()
    ):
        raise ValueError(
            "q025 <= estimated_demand <= q975 does not hold (CONTRACTS.md 2.3)"
        )
    stats = {
        "rows": n,
        "masked_rows": int(base["is_masked"].sum()),
        "stockout_rows": int(base["is_stockout"].sum()),
        "clip_rate": clip_rate,
    }
    return table, stats


def check_recovery(cube: Cube, table: pd.DataFrame, splits_cfg: dict) -> None:
    """Run the `data daily` checks before writing: every real stockout hour from the first training day to the
    last calibration day is filled exactly once, and no other hour is."""
    recovered = recovered_demand(cube, table)  # raises on duplicates and partial days
    final = final_windows(splits_cfg)
    a = cube.day_index(final["train"].start)
    b = cube.day_index(final["calib"].end) + 1
    if np.isnan(recovered[:, a:b]).any():
        raise ValueError("the recovery table leaves stockout days unfilled")


def run(
    model_cfg: dict,
    data_cfg: dict,
    seed: int,
    fold: int | None = None,
    *,
    config_path: str | None = None,
    run_root: str | Path = RUN_ROOT,
) -> dict:
    """Fit, impute every part, write the tables; return the run log."""
    started = time.time()
    model_cls, _ = _architecture(model_cfg)
    levels = quantile_levels(model_cfg)
    splits_cfg = data_cfg["splits"]
    if fold is not None and fold not in fold_windows(splits_cfg):
        raise ValueError(
            f"fold {fold} is not in splits.yaml (folds {sorted(fold_windows(splits_cfg))})"
        )

    mid = model_id(model_cfg)
    rid = f"{mid}-{seed}"
    processed = resolve(data_cfg["paths"]["processed_dir"])
    run_dir = resolve(run_root) / rid / (f"fold{fold}" if fold is not None else "final")
    if fold is not None:
        targets = {"val": processed / "imputation" / rid / f"fold{fold}.parquet"}
    else:
        targets = {
            "calib": processed / "imputation" / rid / "final_calib.parquet",
            "recovery": processed / "imputed_hourly" / f"{rid}.parquet",
        }
    existing = [p for p in [*targets.values(), run_dir / "run_log.json"] if p.exists()]
    if existing:
        raise FileExistsError(
            f"{', '.join(map(str, existing))} exist; imputation outputs are never overwritten"
        )
    masks_path = processed / "eval_masks.parquet"
    if not masks_path.exists():
        raise FileNotFoundError(
            f"{masks_path} is missing: run `python -m uncertainty_aware_planner.data masks` first"
        )
    masks_sha = file_sha256(masks_path)

    cube = load_cube(data_cfg)
    masks = pd.read_parquet(masks_path)
    train_set, _ = build_pypots_set(cube, data_cfg, "train", fold=fold)
    n_train = len(train_set["X"])
    set_seed(seed)
    model = model_cls(model_cfg)
    with warnings.catch_warnings():
        # No val_set on purpose: it would hand the masked hours' true sales to fit (README, Leakage checks).
        warnings.filterwarnings("ignore", message="No val_set given")
        model.fit(train_set)
    del train_set
    model.save(run_dir / "model", overwrite=True)

    outputs = {}
    for part, path in targets.items():
        dataset, index = build_pypots_set(
            cube,
            data_cfg,
            part,
            fold=fold,
            masks=None if part == "recovery" else masks,
        )
        sales = impute_sales(model, dataset)
        table, stats = imputed_table(
            dataset,
            index,
            sales,
            cube,
            levels,
            model_id=mid,
            seed=seed,
            fold=fold,
            split=None if part == "recovery" else part,
            splits_cfg=splits_cfg,
        )
        del dataset, index, sales
        if part == "recovery":
            check_recovery(cube, table, splits_cfg)
        write_parquet(table, path, overwrite=False)
        write_manifest(
            path,
            data_cfg,
            config_path,
            seed,
            {
                "run_id": rid,
                "model_id": mid,
                "fold": fold,
                "part": part,
                "eval_masks_sha256": masks_sha,
                **stats,
            },
        )
        outputs[part] = {"table": str(path), **stats}

    log = {
        "run_id": rid,
        "model_id": mid,
        "architecture": model_cfg["name"],
        "seed": seed,
        "fold": fold,
        "parts": list(targets),
        "git_commit": get_git_state(resolve(".")),
        "model_config_path": config_path,
        "model_config_sha256": file_sha256(resolve(config_path))
        if config_path
        else None,
        "model_config": model_cfg,
        "dataset_revision": data_cfg["source"].get("revision"),
        "splits_config_sha256": file_sha256(resolve(data_cfg["splits_config"])),
        "eval_masks": str(masks_path),
        "eval_masks_sha256": masks_sha,
        "n_train_samples": n_train,
        "model_dir": str(run_dir / "model"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pypots": pypots.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else platform.processor() or platform.machine(),
        "outputs": outputs,
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": round(time.time() - started, 1),
    }
    (run_dir / "run_log.json").write_text(json.dumps(log, indent=2, default=str) + "\n")
    return log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        required=True,
        help="imputation model config, e.g. configs/imputation_models/...yaml",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--fold", type=int, help="rolling fold 1..6; omit for the final model"
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args(argv)

    log = run(
        load_yaml(args.config),
        load_config(args.data_config),
        args.seed,
        args.fold,
        config_path=args.config,
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
