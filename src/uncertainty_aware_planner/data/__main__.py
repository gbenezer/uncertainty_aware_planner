"""Data-stage command line. Every stage takes --config (configs/data.yaml) and --seed.

python -m uncertainty_aware_planner.data load
python -m uncertainty_aware_planner.data splits
python -m uncertainty_aware_planner.data masks [--seed 1024]
python -m uncertainty_aware_planner.data daily --imputed PATH --run-id ID [--edit-log PATH]

Model inputs are built in memory by pypots_arrays.build_pypots_set (imputation) and
timeseries_dataset.build_timeseries_datasets (forecasting), not by a command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config, resolve
from .cube import Cube, load_cube
from .daily import build_demand_daily
from .masks import make_eval_masks
from .outputs import write_manifest, write_parquet
from .splits import splits_table, validate_splits


def _validated_cube(cfg: dict) -> Cube:
    cube = load_cube(cfg)
    validate_splits(
        cfg["splits"], cube.dates, cube.dates[-cfg["expected"]["test_days"] :]
    )
    return cube


def run_load(cfg: dict, args: argparse.Namespace, processed: Path) -> None:
    cube = load_cube(cfg)
    print(
        f"{cube.n_series} series x {cube.n_days} days ({cube.dates[0]} to {cube.dates[-1]}) passed validation"
    )
    print(f"cleaning: {cube.cleaning_report}")


def run_splits(cfg: dict, args: argparse.Namespace, processed: Path) -> None:
    _validated_cube(cfg)
    output = processed / "splits.parquet"
    table = splits_table(cfg["splits"])
    write_parquet(table, output)
    write_manifest(output, cfg, args.config, args.seed, {"rows": len(table)})
    print(table.to_string(index=False))


def run_masks(cfg: dict, args: argparse.Namespace, processed: Path) -> None:
    cube = _validated_cube(cfg)
    masks = make_eval_masks(cube, cfg["splits"], args.seed)
    output = processed / "eval_masks.parquet"
    write_parquet(masks, output)
    counts = masks.groupby(["split", "fold"], dropna=False).size()
    write_manifest(
        output,
        cfg,
        args.config,
        args.seed,
        {"hidden_hours": {str(k): int(v) for k, v in counts.items()}},
    )
    print(f"hidden hours per split/fold:\n{counts.to_string()}")


def run_daily(cfg: dict, args: argparse.Namespace, processed: Path) -> None:
    cube = _validated_cube(cfg)
    imputed = pd.read_parquet(resolve(args.imputed))
    edit_log = pd.read_parquet(resolve(args.edit_log)) if args.edit_log else None
    demand, edit_ids = build_demand_daily(
        cube, imputed, args.run_id, cfg["splits"], edit_log
    )
    output = processed / "demand_daily" / f"{args.run_id}.parquet"
    write_parquet(demand, output, overwrite=False)
    details = {
        "imputed_hourly": args.imputed,
        "rows": len(demand),
        "edit_ids": edit_ids,
    }
    details["days_without_recovery"] = int(demand["demand_recovered"].isna().sum())
    write_manifest(output, cfg, args.config, args.seed, details)
    print(f"wrote {output}: {len(demand)} rows, {len(edit_ids)} edits applied")


STAGES = {
    "load": run_load,
    "splits": run_splits,
    "masks": run_masks,
    "daily": run_daily,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m uncertainty_aware_planner.data")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--seed", type=int, default=None, help="mask seed; defaults to splits.yaml"
    )
    parser.add_argument("--imputed", help="imputed_hourly parquet (daily stage)")
    parser.add_argument("--run-id", help="imputation run id (daily stage)")
    parser.add_argument("--edit-log", help="edit_log parquet (daily stage, optional)")
    args = parser.parse_args(argv)
    if args.stage == "daily" and not (args.imputed and args.run_id):
        parser.error("daily needs --imputed and --run-id")

    cfg = load_config(args.config)
    STAGES[args.stage](cfg, args, resolve(cfg["paths"]["processed_dir"]))


if __name__ == "__main__":
    main()
