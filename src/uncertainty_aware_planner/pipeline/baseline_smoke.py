"""Smoke test of the imputation baseline on the SYNTHETIC fixtures: one command, CPU, no credentials.

    uv run python -m uncertainty_aware_planner.pipeline.baseline_smoke [--record]

The baseline is the non-quantile DLinear from the dataset paper repository (Feature Cards 1.1). This runs the
same code path as the full-data run, experiments/gil/data_pipeline_experimentation.py: the same model config,
rolling folds, eval masks and metric (evaluation/error_metrics.py), with each model built right after
set_seed. Two variants per (seed, fold), as in that script:
  not_leaked  fit on the fold's train window only (the official number)
  leaked      fit on the val window's visible hours (as the dataset authors train on all days), for reference

Before training it checks the environment (Python version), the data (fixture SHA-256 against
configs/baseline_smoke.yaml) and the split (configs/splits.yaml validated against the data calendar). After
training it compares mean WAPE and WPE with the smoke reference row in results/imputation/baseline_log.csv
and exits with status 1 on any mismatch. Every run writes a run log to data/runs/baseline_smoke/ (gitignored);
--record also appends this run's rows to the committed log (needs a clean working tree).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pypots
import torch
from pypots.imputation.dlinear import DLinear
from pypots.optim.adam import Adam
from pypots.utils.logging import logger as pypots_logger

from uncertainty_aware_planner.data.config import load_config, load_yaml, resolve
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.masks import make_eval_masks
from uncertainty_aware_planner.data.outputs import file_sha256
from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set
from uncertainty_aware_planner.data.splits import validate_splits
from uncertainty_aware_planner.evaluation.error_metrics import (
    calculate_imputation_percent_errors,
)
from uncertainty_aware_planner.utils.util_functions import set_seed

DEFAULT_CONFIG = "configs/baseline_smoke.yaml"
VARIANTS = ("not_leaked", "leaked")
LOG_COLUMNS = [
    "date_utc",
    "run",
    "model",
    "variant",
    "data",
    "data_revision",
    "data_sha256",
    "split_config_sha256",
    "commit",
    "model_config",
    "model_config_sha256",
    "seeds",
    "folds",
    "n_runs",
    "hardware",
    "wape_mean",
    "wape_sd",
    "wpe_mean",
    "wpe_sd",
    "source",
    "notes",
]


class CheckFailed(RuntimeError):
    pass


class _DropSavingPathWarning(logging.Filter):
    """PyPOTS warns once per model that saving_path is None (on purpose here: nothing is saved).

    A filter, because PyPOTS resets its logger level whenever a model is built (pypots/base.py).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "saving_path not given" not in record.getMessage()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=resolve(".")
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> tuple[str | None, bool]:
    """(HEAD commit, whether tracked files have uncommitted changes)."""
    return _git("rev-parse", "HEAD"), bool(
        _git("status", "--porcelain", "--untracked-files=no")
    )


def hardware() -> str:
    """CPU model where the OS reports one; results are only compared on CPU runs."""
    name = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            name = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except (OSError, subprocess.CalledProcessError):
        pass
    return (
        f"{name}, {os.cpu_count()} cores, {platform.system()} {platform.machine()}, CPU"
    )


def check_environment() -> dict:
    expected = resolve(".python-version").read_text().strip()
    found = platform.python_version()
    if found != expected:
        raise CheckFailed(
            f"Python {found} is running but .python-version pins {expected}; "
            "run through `uv run` so uv uses the pinned interpreter"
        )
    return {
        "python": found,
        "python_pinned": expected,
        "uv_lock_sha256": file_sha256(resolve("uv.lock")),
        "torch": torch.__version__,
        "pypots": pypots.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "hardware": hardware(),
    }


def check_data(source: dict) -> dict[str, dict]:
    found = {}
    for name, expected in source["files"].items():
        path = resolve(name)
        found[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        if found[name] != expected:
            raise CheckFailed(
                f"{name} is {found[name]}, but configs/baseline_smoke.yaml expects {expected}"
            )
    return found


def data_cfg_for_fixtures(smoke: dict, work_dir: Path) -> dict:
    """configs/data.yaml (with splits.yaml) pointed at the fixtures; outputs and caches go to work_dir."""
    cfg = load_config(smoke["data_config"])
    cfg["source"] = {
        "kind": smoke["source"]["kind"],
        "data_files": smoke["source"]["data_files"],
    }
    cfg["expected"] = smoke["expected"]
    cfg["paths"] = {
        "cache_dir": str(work_dir / "hf_cache"),
        "raw_dir": str(work_dir / "raw"),
        "processed_dir": str(work_dir / "processed"),
    }
    return cfg


def make_dlinear(model_cfg: dict, device: str) -> DLinear:
    """Built exactly as make_dlinear() in experiments/gil/data_pipeline_experimentation.py."""
    return DLinear(
        n_steps=model_cfg["n_steps"],
        n_features=model_cfg["n_features"],
        moving_avg_window_size=model_cfg["moving_avg_window_size"],
        individual=False,
        d_model=model_cfg["d_model"],
        ORT_weight=model_cfg["ORT_weight"],
        MIT_weight=model_cfg["MIT_weight"],
        batch_size=model_cfg["batch_size"],
        epochs=model_cfg["epochs"],
        patience=model_cfg["patience"],
        optimizer=Adam(**model_cfg["optimizer_parameters"]),
        num_workers=0,
        device=torch.device(device),
        saving_path=None,
        model_saving_strategy="best",
        verbose=False,
    )


def train_and_score(
    cube, cfg: dict, masks: pd.DataFrame, model_cfg: dict, smoke: dict
) -> list[dict]:
    rows = []
    for seed in smoke["seeds"]:
        for fold in smoke["folds"]:
            train_set, _ = build_pypots_set(cube, cfg, "train", fold=fold)
            val_set, _ = build_pypots_set(cube, cfg, "val", fold=fold, masks=masks)
            fit_sets = {"not_leaked": train_set, "leaked": {"X": val_set["X"]}}
            for variant in VARIANTS:
                set_seed(seed)
                model = make_dlinear(model_cfg, smoke["device"])
                model.fit(fit_sets[variant])
                imputation = model.predict(val_set)["imputation"]
                result = calculate_imputation_percent_errors(
                    eval_pypots=val_set, imputation=imputation, daily_pool=True
                )
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "variant": variant,
                        "wape": float(result["wape"]),
                        "wpe": float(result["wpe"]),
                    }
                )
    return rows


def summarise(rows: list[dict]) -> dict[str, dict]:
    df = pd.DataFrame(rows)
    out = {}
    for variant, group in df.groupby("variant", sort=False):
        out[variant] = {
            "n_runs": len(group),
            "wape_mean": group["wape"].mean(),
            "wape_sd": group["wape"].std(),
            "wpe_mean": group["wpe"].mean(),
            "wpe_sd": group["wpe"].std(),
        }
    return out


def reference_rows(log_path: Path) -> dict[str, dict]:
    """The last committed smoke row per variant."""
    if not log_path.exists():
        return {}
    with open(log_path, newline="") as f:
        return {
            row["variant"]: row for row in csv.DictReader(f) if row["run"] == "smoke"
        }


def compare(summary: dict, references: dict, data_sha: str, tol: float) -> list[dict]:
    checks = []
    for variant, stats in summary.items():
        ref = references.get(variant)
        if ref is None:
            checks.append({"variant": variant, "status": "NO REFERENCE"})
            continue
        diff = {
            key: abs(stats[key] - float(ref[key])) for key in ("wape_mean", "wpe_mean")
        }
        same_data = ref["data_sha256"] == data_sha
        ok = same_data and all(d <= tol for d in diff.values())
        checks.append(
            {
                "variant": variant,
                "status": "MATCH" if ok else "MISMATCH",
                "reference_commit": ref["commit"],
                "reference_wape_mean": float(ref["wape_mean"]),
                "reference_wpe_mean": float(ref["wpe_mean"]),
                "abs_diff": diff,
                "same_data_sha256": same_data,
            }
        )
    return checks


def append_to_log(log_path: Path, record: dict) -> None:
    new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            writer.writeheader()
        for variant, stats in record["summary"].items():
            writer.writerow(
                {
                    "date_utc": record["finished_at_utc"][:10],
                    "run": "smoke",
                    "model": "dlinear-imputation-baseline",
                    "variant": variant,
                    "data": "SYNTHETIC fixtures (fixtures/raw_series)",
                    "data_revision": "",
                    "data_sha256": record["data_sha256"],
                    "split_config_sha256": record["split"]["sha256"],
                    "commit": record["git_commit"],
                    "model_config": record["model_config"]["path"],
                    "model_config_sha256": record["model_config"]["sha256"],
                    "seeds": " ".join(map(str, record["seeds"])),
                    "folds": " ".join(map(str, record["folds"])),
                    "n_runs": stats["n_runs"],
                    "hardware": record["environment"]["hardware"],
                    "wape_mean": f"{stats['wape_mean']:.6f}",
                    "wape_sd": f"{stats['wape_sd']:.6f}",
                    "wpe_mean": f"{stats['wpe_mean']:.6f}",
                    "wpe_sd": f"{stats['wpe_sd']:.6f}",
                    "source": "uv run python -m uncertainty_aware_planner.pipeline.baseline_smoke",
                    "notes": f"Python {record['environment']['python']}, torch "
                    f"{record['environment']['torch']}, pypots {record['environment']['pypots']}",
                }
            )


def run(config_path: str, record: bool) -> int:
    started = time.time()
    smoke = load_yaml(config_path)
    commit, dirty = git_state()
    if record and dirty:
        raise CheckFailed(
            "--record needs a clean working tree, so the logged commit is the code that ran"
        )

    environment = check_environment()
    print(
        f"[check] environment  Python {environment['python']} matches .python-version; "
        f"torch {environment['torch']}, pypots {environment['pypots']}; "
        f"uv.lock sha256 {environment['uv_lock_sha256'][:12]}"
    )
    data_files = check_data(smoke["source"])
    data_sha = ";".join(
        f"{split}={data_files[path]['sha256']}"
        for split, path in smoke["source"]["data_files"].items()
    )
    for path, found in data_files.items():
        print(
            f"[check] data         {path} sha256 {found['sha256'][:12]} matches {config_path}"
        )

    work_dir = resolve(smoke["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    model_cfg_path = smoke["model_config"]
    model_cfg = load_yaml(model_cfg_path)
    torch.use_deterministic_algorithms(True)
    pypots_logger.addFilter(_DropSavingPathWarning())

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        cfg = data_cfg_for_fixtures(smoke, Path(tmp))
        cube = load_cube(cfg)
        test_days = cfg["expected"]["test_days"]
        validate_splits(cfg["splits"], cube.dates, cube.dates[-test_days:])
        split_sha = file_sha256(resolve(cfg["splits_config"]))
        print(
            f"[check] split        {cfg['splits_config']} sha256 {split_sha[:12]}: "
            f"{len(cfg['splits']['folds']['cutoffs'])} folds validated against the "
            f"{cube.n_days}-day calendar"
        )
        print(
            f"[check] seeds        set_seed() before every model build: {smoke['seeds']}"
        )
        masks = make_eval_masks(cube, cfg["splits"])
        rows = train_and_score(cube, cfg, masks, model_cfg, smoke)

    summary = summarise(rows)
    log_path = resolve(smoke["log"])
    checks = compare(summary, reference_rows(log_path), data_sha, smoke["tolerance"])
    finished = datetime.now(UTC)
    run_record = {
        "run": "smoke",
        "finished_at_utc": finished.isoformat(timespec="seconds"),
        "wall_seconds": round(time.time() - started, 1),
        "git_commit": commit,
        "git_dirty": dirty,
        "config": config_path,
        "environment": environment,
        "data_files": data_files,
        "data_sha256": data_sha,
        "split": {
            "config": cfg["splits_config"],
            "sha256": split_sha,
            "cutoffs": [str(c) for c in cfg["splits"]["folds"]["cutoffs"]],
        },
        "model_config": {
            "path": model_cfg_path,
            "sha256": file_sha256(resolve(model_cfg_path)),
            "values": model_cfg,
        },
        "device": smoke["device"],
        "seeds": smoke["seeds"],
        "folds": smoke["folds"],
        "results": rows,
        "summary": summary,
        "reference_checks": checks,
    }
    run_log = work_dir / f"run_log_{finished.strftime('%Y%m%dT%H%M%SZ')}.json"
    run_log.write_text(json.dumps(run_record, indent=2, default=str) + "\n")

    print(
        "\nImputation baseline, SYNTHETIC fixtures (a pipeline check, not the official number)"
    )
    print(
        f"{'variant':<11}{'runs':>5}  {'WAPE mean ± sd':<16} {'WPE mean ± sd':<17} "
        f"{'reference WAPE':<26} status"
    )
    for check in checks:
        s = summary[check["variant"]]
        ref = (
            f"{check['reference_wape_mean']:.4f} (commit {check['reference_commit'][:7]})"
            if "reference_wape_mean" in check
            else "-"
        )
        wape = f"{s['wape_mean']:.4f} ± {s['wape_sd']:.4f}"
        wpe = f"{s['wpe_mean']:+.4f} ± {s['wpe_sd']:.4f}"
        print(
            f"{check['variant']:<11}{s['n_runs']:>5}  {wape:<16} {wpe:<17} {ref:<26} "
            f"{check['status']}"
        )
    print(f"\nrun log: {run_log.relative_to(resolve('.'))}")
    print(f"wall time: {run_record['wall_seconds']} s")
    print(f"official full-data baseline: {smoke['log']} (rows with run=full)")

    if record:
        append_to_log(log_path, run_record)
        print(f"recorded smoke rows in {smoke['log']} for commit {commit}")
        return 0
    if any(c["status"] == "MISMATCH" for c in checks):
        print(
            f"MISMATCH: mean WAPE or WPE differs from the reference by more than "
            f"{smoke['tolerance']}, or the fixture data changed",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--record",
        action="store_true",
        help="append this run's rows to the committed log (clean working tree only)",
    )
    args = parser.parse_args(argv)
    try:
        sys.exit(run(args.config, args.record))
    except CheckFailed as error:
        print(f"[check] FAILED: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
