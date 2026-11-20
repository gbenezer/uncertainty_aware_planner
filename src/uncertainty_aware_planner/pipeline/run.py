"""One command runs a forecasting experiment end to end: data, imputation, demand_daily, forecasts, metrics.

    python -m uncertainty_aware_planner.pipeline.run --experiment configs/experiments/<name>.yaml \
        [--stages data impute daily forecast evaluate] [--seeds 2025 ...] [--folds 1 ...] \
        [--force] [--metrics-only] [--dry-run] [--smoke [--work-dir DIR]]

An experiment file names one imputation run, one forecasting model and how to score it:

    name: dlinear_baseline_censored
    data_config: configs/data.yaml
    imputation:
      run_id: imp-none-censored
      source: censored        # censored: built here with pipeline.censored; imputed:
                              # <processed>/imputed_hourly/<run_id>.parquet must exist (written by impute.py)
    forecast:
      config: configs/forecasting_models/dlinear_forecasting_baseline.yaml
      seeds: [2025, 2026, 2027]
      folds: [1, 2, 3, 4, 5, 6]
      final_parts: [calib, test]
      protocol: project       # project | paper (paper: train through the calibration week, test only,
                              # *_paper_split folders, no folds)
    evaluate:
      point: estimated_demand # or q500
      truth: data/processed/demand_daily/imp-none-censored.parquet
      slices: data/processed/slices.parquet
      out: results/forecasting

Stages run in process, in this order, and each is skipped when its outputs exist:
    data      data load, unless <raw_dir>/DATA_VERSION.json exists
    impute    censored: write <processed>/imputed_hourly/<run_id>.parquet; imputed: require that file
    daily     data daily for <run_id>, unless <processed>/demand_daily/<run_id>.parquet exists
    forecast  per seed, the fold runs then the final run; a run is skipped when its tables and their
              manifests exist, unless --force
    evaluate  pipeline.evaluate_forecasting on this experiment's tables, unless its metrics already cover
              them with the same settings (a forecast run in this invocation or --force re-evaluates)
--force applies to forecast and evaluate only: imputed_hourly and demand_daily files are versioned and never
overwritten. --metrics-only deletes the fold tables (and their manifests) after a successful evaluation;
finals, run directories and metrics stay, and later invocations count those fold runs as done.

--smoke runs the whole chain on the SYNTHETIC fixtures with the model's smoke settings, the first seed and
the first fold, under --work-dir (default data/runs/pipeline_smoke/<name>); the smoke demand_daily is also
the truth. An imputed source gets the censored stand-in there, since a smoke run only checks the code path.

The experiment log goes to <out>/<run_id>/<protocol>/experiment_<name>.json: the experiment, git commit,
status and wall time of every stage and run, the tables, and the last evaluation.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from uncertainty_aware_planner.data.config import (
    REPO_ROOT,
    load_config,
    load_yaml,
    resolve,
)

STAGES = ("data", "impute", "daily", "forecast", "evaluate")
SOURCES = ("censored", "imputed")
PROTOCOLS = ("project", "paper")
POINTS = ("estimated_demand", "q500")
FINAL_PARTS = ("calib", "test")
PAPER_SUFFIX = "_paper_split"
SMOKE_WORK_DIR = "data/runs/pipeline_smoke"
PRUNED = "fold table deleted by --metrics-only after evaluation; run log kept"
# Looked up when used, not at import: tests swap them, and the evaluation CLI may not exist yet.
FORECAST_MODULE = "uncertainty_aware_planner.pipeline.forecast"
EVALUATE_MODULE = "uncertainty_aware_planner.pipeline.evaluate_forecasting"


class ExperimentError(RuntimeError):
    """A malformed experiment or a missing input; main prints it without a traceback."""


@dataclass
class Run:
    """One forecast.run call: a fold run (eval part "val") or the final run (its final parts)."""

    seed: int
    fold: int | None
    parts: list[str]
    tables: list[Path]
    run_dir: Path

    @property
    def label(self) -> str:
        return f"seed {self.seed} " + (
            f"fold {self.fold}" if self.fold is not None else "final"
        )


@dataclass
class Setup:
    """Everything one invocation reads and writes, resolved once from the experiment (and --smoke)."""

    name: str
    run_id: str
    source: str
    protocol: str
    point: str
    data_cfg: dict
    data_config: str  # what the data-stage manifests record as the config
    run_data_cfg: dict  # as forecast runs see it (paper: train through calib)
    model_cfg: dict
    seeds: list[int]
    runs: list[Run]
    table_dirs: list[Path]  # one per seed
    truth: Path
    slices: Path
    out: Path
    work_dir: Path | None  # --smoke only

    @property
    def smoke(self) -> bool:
        return self.work_dir is not None

    @property
    def processed(self) -> Path:
        return resolve(self.data_cfg["paths"]["processed_dir"])

    @property
    def imputed_path(self) -> Path:
        return self.processed / "imputed_hourly" / f"{self.run_id}.parquet"

    @property
    def demand_path(self) -> Path:
        return self.processed / "demand_daily" / f"{self.run_id}.parquet"

    @property
    def quantiles_root(self) -> Path:
        return resolve(self.model_cfg["run_dir"])

    @property
    def eval_dir(self) -> Path:
        return self.out / self.run_id / self.protocol

    @property
    def record_path(self) -> Path:
        return self.eval_dir / f"experiment_{self.name}.json"


def _show(path: Path | str) -> str:
    """Repo-relative inside the repo, so logs and plans read the same on every machine."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _shown_argv(argv: list[str]) -> list[str]:
    prefix = f"{REPO_ROOT}{os.sep}"
    return [arg.removeprefix(prefix) for arg in argv]


def _manifest(table: Path) -> Path:
    return table.with_name(table.name + ".manifest.json")


def _first_missing(tables: list[Path]) -> Path | None:
    for table in tables:
        for path in (table, _manifest(table)):
            if not path.exists():
                return path
    return None


def _git_state() -> tuple[str | None, bool | None]:
    """(HEAD commit, whether tracked files have uncommitted changes)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(
            git("status", "--porcelain", "--untracked-files=no")
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _say(verb: str, stage: str, reason: str) -> dict:
    print(f"{verb} {stage}: {reason}")
    return {"status": "skipped" if verb == "skip" else "ran", "reason": reason}


def _checked(experiment: dict) -> dict:
    """A validated copy, with forecast.folds/final_parts/protocol and evaluate.point defaulted."""
    if not isinstance(experiment, dict):
        raise ExperimentError("an experiment must be a YAML mapping")
    exp = copy.deepcopy(experiment)
    for section in ("imputation", "forecast", "evaluate"):
        if not isinstance(exp.get(section), dict):
            raise ExperimentError(f"the experiment needs an '{section}:' section")
    fc, ev = exp["forecast"], exp["evaluate"]
    fc.setdefault("folds", [])
    fc.setdefault("final_parts", list(FINAL_PARTS))
    fc.setdefault("protocol", "project")
    ev.setdefault("point", "estimated_demand")
    required = {
        "imputation": ("run_id", "source"),
        "forecast": ("config", "seeds"),
        "evaluate": ("truth", "slices", "out"),
    }
    missing = [
        f"{s}.{k}" for s, keys in required.items() for k in keys if k not in exp[s]
    ]
    missing = ([] if exp.get("name") else ["name"]) + missing
    if missing:
        raise ExperimentError(f"the experiment is missing {', '.join(missing)}")
    for section, key, allowed in (
        ("imputation", "source", SOURCES),
        ("forecast", "protocol", PROTOCOLS),
        ("evaluate", "point", POINTS),
    ):
        if exp[section][key] not in allowed:
            raise ExperimentError(
                f"{section}.{key} must be one of {', '.join(allowed)}, not {exp[section][key]!r}"
            )
    if not fc["seeds"]:
        raise ExperimentError("forecast.seeds is empty")
    if not fc["final_parts"] or set(fc["final_parts"]) - set(FINAL_PARTS):
        raise ExperimentError(f"forecast.final_parts must be a subset of {FINAL_PARTS}")

    from uncertainty_aware_planner.pipeline.censored import RUN_ID as CENSORED

    run_id = exp["imputation"]["run_id"]
    if exp["imputation"]["source"] == "censored" and run_id != CENSORED:
        raise ExperimentError(
            f"source censored is imputation run {CENSORED}; imputation.run_id is {run_id!r}"
        )
    return exp


def _fixture_config(work: Path) -> dict:
    """fixtures/ sits at the repo root, outside the package (`python -m` from the root and pytest find it)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from fixtures.synthetic_raw_series import fixture_config

    return fixture_config(work)


def _setup(experiment: dict, smoke: bool, work_dir: str | Path | None) -> Setup:
    imputation, fc, ev = (experiment[k] for k in ("imputation", "forecast", "evaluate"))
    run_id = imputation["run_id"]
    model_cfg = load_yaml(fc["config"])
    if not isinstance(model_cfg, dict):
        raise ExperimentError(f"{fc['config']} is not a model config")
    absent = [k for k in ("model_id", "output_dir", "run_dir") if k not in model_cfg]
    if absent:
        raise ExperimentError(f"{fc['config']} has no {', '.join(absent)}")
    model_cfg["config_path"] = fc["config"]  # recorded in the table manifests
    seeds, folds, parts = list(fc["seeds"]), list(fc["folds"]), list(fc["final_parts"])
    forecast = importlib.import_module(FORECAST_MODULE)

    work = None
    if smoke:
        work = resolve(work_dir or Path(SMOKE_WORK_DIR) / experiment["name"])
        data_cfg = _fixture_config(work)
        data_config = "fixtures/synthetic_raw_series.py (fixture_config)"
        model_cfg = forecast.smoke_config(model_cfg)
        model_cfg["output_dir"] = str(
            work / "processed" / Path(model_cfg["output_dir"]).name
        )
        model_cfg["run_dir"] = str(work / "runs" / Path(model_cfg["run_dir"]).name)
        processed = resolve(data_cfg["paths"]["processed_dir"])
        truth = processed / "demand_daily" / f"{run_id}.parquet"
        slices = processed / Path(ev["slices"]).name
        out = work / "results" / Path(ev["out"]).name
        seeds, folds = seeds[:1], folds[:1]
    else:
        data_config = experiment.get("data_config") or model_cfg.get("data_config")
        if not data_config:
            raise ExperimentError(
                "no data_config in the experiment or the model config"
            )
        data_cfg = load_config(data_config)
        truth, slices, out = (resolve(ev[k]) for k in ("truth", "slices", "out"))

    run_data_cfg = data_cfg
    if fc["protocol"] == "paper":
        # As experiments/vishnu/paper_split_run.py: the dataset authors' split, in separate folders.
        model_cfg["output_dir"] = str(model_cfg["output_dir"]) + PAPER_SUFFIX
        model_cfg["run_dir"] = str(model_cfg["run_dir"]) + PAPER_SUFFIX
        run_data_cfg = copy.deepcopy(data_cfg)
        final = run_data_cfg["splits"]["final"]
        final["train"] = [final["train"][0], final["calib"][1]]
        folds, parts = [], ["test"]

    from uncertainty_aware_planner.models.forecasting.forecast_table import (
        run_id as forecast_run_id,
    )

    runs, table_dirs = [], []
    for seed in seeds:
        rid = forecast_run_id(model_cfg["model_id"], seed)
        for fold in [*folds, None]:
            run_dir, table_dir = forecast.run_paths(model_cfg, run_id, rid, fold)
            names = (
                [f"fold{fold}"] if fold is not None else [f"final_{p}" for p in parts]
            )
            fold_parts = ["val"] if fold is not None else parts
            tables = [table_dir / f"{n}.parquet" for n in names]
            runs.append(Run(seed, fold, fold_parts, tables, run_dir))
        table_dirs.append(table_dir)

    return Setup(
        name=experiment["name"],
        run_id=run_id,
        source=imputation["source"],
        protocol=fc["protocol"],
        point=ev["point"],
        data_cfg=data_cfg,
        data_config=str(data_config),
        run_data_cfg=run_data_cfg,
        model_cfg=model_cfg,
        seeds=seeds,
        runs=runs,
        table_dirs=table_dirs,
        truth=truth,
        slices=slices,
        out=out,
        work_dir=work,
    )


def _data_args(setup: Setup) -> argparse.Namespace:
    """The arguments the data-stage functions in data/__main__.py read."""
    return argparse.Namespace(
        config=setup.data_config,
        seed=None,
        imputed=_show(setup.imputed_path),
        run_id=setup.run_id,
        edit_log=None,
    )


def _stage_data(setup: Setup, log: dict, *, dry_run: bool, force: bool) -> dict:
    version = resolve(setup.data_cfg["paths"]["raw_dir"]) / "DATA_VERSION.json"
    if version.exists():
        return _say("skip", "data", f"{_show(version)} exists")
    if setup.data_cfg["source"]["kind"] == "hub":
        result = _say("run", "data", f"{_show(version)} missing: data load")
    else:
        result = _say("run", "data", "local parquet source: data load validates it")
    if not dry_run:
        from uncertainty_aware_planner.data.__main__ import STAGES as DATA_STAGES

        DATA_STAGES["load"](setup.data_cfg, _data_args(setup), setup.processed)
    return result


def _stage_impute(setup: Setup, log: dict, *, dry_run: bool, force: bool) -> dict:
    path = setup.imputed_path
    if path.exists():
        return _say("skip", "impute", f"{_show(path)} exists")
    if setup.source == "imputed" and not setup.smoke:
        raise ExperimentError(
            f"{_show(path)} does not exist. Imputation source 'imputed' reads the imputed_hourly table "
            f"that pipeline/impute.py writes for run {setup.run_id}: run pipeline/impute.py first, or "
            "make imputation.run_id match the run id it writes"
        )
    note = " (smoke stand-in for an imputed run)" if setup.source == "imputed" else ""
    result = _say("run", "impute", f"{_show(path)} missing: censored run{note}")
    if not dry_run:
        from uncertainty_aware_planner.data.cube import load_cube
        from uncertainty_aware_planner.pipeline.censored import censored_imputed_hourly

        rows = censored_imputed_hourly(load_cube(setup.data_cfg))
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False)
        print(f"wrote {_show(path)}: {len(rows)} stockout hours")
    return result


def _stage_daily(setup: Setup, log: dict, *, dry_run: bool, force: bool) -> dict:
    path = setup.demand_path
    if path.exists():
        return _say("skip", "daily", f"{_show(path)} exists")
    result = _say("run", "daily", f"{_show(path)} missing: data daily")
    if not dry_run:
        if not setup.imputed_path.exists():
            raise ExperimentError(
                f"{_show(setup.imputed_path)} does not exist: run the impute stage first"
            )
        from uncertainty_aware_planner.data.__main__ import STAGES as DATA_STAGES

        DATA_STAGES["daily"](setup.data_cfg, _data_args(setup), setup.processed)
    return result


def _run_needed(run: Run, pruned: set[str], force: bool) -> tuple[bool, str]:
    if force:
        return True, "--force"
    missing = _first_missing(run.tables)
    if missing is None:
        return False, "tables exist"
    if (
        run.fold is not None
        and (run.run_dir / "run_log.json").exists()
        and all(_show(t) in pruned for t in run.tables)
    ):
        return False, PRUNED
    return True, f"{_show(missing)} missing"


def _stage_forecast(setup: Setup, log: dict, *, dry_run: bool, force: bool) -> dict:
    pruned = set(log["pruned_tables"])
    todo = []
    for run in setup.runs:
        needed, reason = _run_needed(run, pruned, force)
        entry = {
            "seed": run.seed,
            "fold": run.fold,
            "parts": run.parts,
            "tables": [_show(t) for t in run.tables],
            "status": ("planned" if dry_run else "pending") if needed else "skipped",
            "reason": reason,
        }
        log["runs"].append(entry)
        if needed:
            todo.append((run, entry))
    what = f"{setup.model_cfg['model_id']}, {setup.protocol} protocol"
    if not todo:
        n_pruned = sum(entry["reason"] == PRUNED for entry in log["runs"])
        if n_pruned:
            what += f"; {n_pruned} with the fold table deleted by --metrics-only"
        return _say("skip", "forecast", f"all {len(setup.runs)} runs done ({what})")
    result = _say("run", "forecast", f"{len(todo)} of {len(setup.runs)} runs ({what})")
    for run, entry in zip(setup.runs, log["runs"], strict=True):
        verb = "skip" if entry["status"] == "skipped" else "run"
        print(f"  {verb} {run.label}: {entry['reason']}")
    if dry_run:
        return result

    if not setup.demand_path.exists():
        raise ExperimentError(
            f"{_show(setup.demand_path)} does not exist: run the daily stage first"
        )
    demand = pd.read_parquet(setup.demand_path)
    found = sorted(demand["imputation_run_id"].astype(str).unique())
    if found != [setup.run_id]:
        raise ExperimentError(
            f"{_show(setup.demand_path)} holds imputation run {found}, not {setup.run_id}"
        )
    forecast = importlib.import_module(FORECAST_MODULE)
    for run, entry in todo:
        started = time.time()
        entry["status"] = "failed"
        print(f"  start {run.label}")
        forecast.run(
            setup.model_cfg,
            setup.run_data_cfg,
            demand,
            run.seed,
            fold=run.fold,
            eval_parts=None if run.fold is not None else list(run.parts),
            demand_daily_path=_show(setup.demand_path),
        )
        missing = _first_missing(run.tables)
        if missing is not None:
            raise RuntimeError(
                f"{run.label} finished without {missing}: pipeline/run.py and pipeline/forecast.py "
                "disagree on the output layout"
            )
        entry["status"] = "ran"
        entry["wall_seconds"] = round(time.time() - started, 1)
        print(f"  done {run.label} in {entry['wall_seconds']} s")
    return result


def _evaluate_argv(setup: Setup) -> list[str]:
    argv = [
        "--tables",
        *(str(d / "*.parquet") for d in setup.table_dirs),
        "--truth",
        str(setup.truth),
        "--slices",
        str(setup.slices),
        "--point",
        setup.point,
        "--quantiles-root",
        str(setup.quantiles_root),
        "--protocol",
        setup.protocol,
        "--out",
        str(setup.out),
    ]
    return argv + (["--no-ledger"] if setup.smoke else [])


def _evaluation_needed(
    setup: Setup, log: dict, argv: list[str], force: bool
) -> tuple[bool, str]:
    if force:
        return True, "--force"
    forecast = log["stages"].get("forecast", {}).get("status")
    if forecast in ("ran", "planned"):
        return True, f"forecast runs {forecast}"
    metrics = setup.eval_dir / "metrics.csv"
    if not metrics.exists():
        return True, f"{_show(metrics)} missing"
    if (log["evaluation"] or {}).get("argv") != _shown_argv(argv):
        return (
            True,
            f"{_show(setup.record_path)} records no evaluation with these settings",
        )
    tables = [t for d in setup.table_dirs for t in d.glob("*.parquet")]
    if any(t.stat().st_mtime > metrics.stat().st_mtime for t in tables):
        return True, f"a forecast table is newer than {_show(metrics)}"
    return False, f"{_show(metrics)} covers these tables"


def _stage_evaluate(setup: Setup, log: dict, *, dry_run: bool, force: bool) -> dict:
    argv = _evaluate_argv(setup)
    needed, reason = _evaluation_needed(setup, log, argv, force)
    if not needed:
        return _say("skip", "evaluate", reason)
    result = _say("run", "evaluate", reason)
    result["argv"] = _shown_argv(argv)
    if dry_run:
        return result
    empty = [d for d in setup.table_dirs if not any(d.glob("*.parquet"))]
    if empty:
        raise ExperimentError(
            f"no forecast tables in {', '.join(_show(d) for d in empty)}: run the forecast "
            "stage first, or leave those seeds out with --seeds"
        )
    if not setup.truth.exists():
        raise ExperimentError(
            f"the evaluation truth {_show(setup.truth)} does not exist"
        )
    evaluate = importlib.import_module(EVALUATE_MODULE)
    try:
        code = evaluate.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    if code not in (0, None):
        raise RuntimeError(f"pipeline.evaluate_forecasting exited with status {code}")
    log["evaluation"] = {
        "argv": result["argv"],
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": _show(setup.eval_dir / "metrics.csv"),
    }
    return result


HANDLERS = {
    "data": _stage_data,
    "impute": _stage_impute,
    "daily": _stage_daily,
    "forecast": _stage_forecast,
    "evaluate": _stage_evaluate,
}


def _prune_fold_tables(setup: Setup, log: dict, dry_run: bool) -> dict:
    """--metrics-only: delete this experiment's fold tables once metrics cover them."""
    if dry_run:
        print("metrics-only: fold tables are deleted after a successful evaluation")
        return {"status": "planned"}
    metrics = setup.eval_dir / "metrics.csv"
    if log["stages"]["evaluate"]["status"] not in ("ran", "skipped"):
        reason = "no evaluation in this invocation"
    elif not metrics.exists():
        reason = f"{_show(metrics)} missing"
    else:
        reason = None
    if reason:
        print(f"metrics-only: fold tables kept ({reason})")
        return {"status": "skipped", "reason": reason}
    removed = []
    for table_dir in setup.table_dirs:
        for table in sorted(table_dir.glob("fold*.parquet")):
            table.unlink()
            _manifest(table).unlink(missing_ok=True)
            removed.append(_show(table))
    log["pruned_tables"] = sorted(set(log["pruned_tables"]) | set(removed))
    print(
        f"metrics-only: fold tables deleted: {len(removed)}; metrics: {_show(metrics)}"
    )
    return {"status": "ran", "deleted": removed}


def run_experiment(
    experiment: dict,
    *,
    stages: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    smoke: bool = False,
    metrics_only: bool = False,
    work_dir: str | Path | None = None,
) -> dict:
    """Run (dry_run: only print) the requested stages of one experiment; return the experiment log.

    Unless dry_run, the log is written to <out>/<run_id>/<protocol>/experiment_<name>.json, also when a
    stage fails. The previous log there supplies the last evaluation and the fold tables --metrics-only
    deleted.
    """
    started = time.time()
    experiment = _checked(experiment)
    requested = list(STAGES) if stages is None else list(stages)
    unknown = sorted(set(requested) - set(STAGES))
    if unknown:
        raise ExperimentError(
            f"unknown stages {unknown}; choose from {', '.join(STAGES)}"
        )
    setup = _setup(experiment, smoke, work_dir)
    previous = (
        json.loads(setup.record_path.read_text()) if setup.record_path.exists() else {}
    )
    commit, dirty = _git_state()
    log = {
        "name": setup.name,
        "experiment": experiment,
        "options": {
            "stages": [s for s in STAGES if s in requested],
            "force": force,
            "smoke": smoke,
            "metrics_only": metrics_only,
            "work_dir": _show(setup.work_dir) if setup.work_dir else None,
        },
        "git_commit": commit,
        "git_dirty": dirty,
        "started_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "stages": {},
        "runs": [],
        "tables": [_show(t) for run in setup.runs for t in run.tables],
        "evaluation": previous.get("evaluation"),
        "pruned_tables": previous.get("pruned_tables", []),
    }

    folds = sorted({r.fold for r in setup.runs if r.fold is not None})
    print(
        f"experiment {setup.name}: {setup.model_cfg['model_id']} on {setup.run_id} ({setup.source}), "
        f"{setup.protocol} protocol, seeds {setup.seeds}, folds {folds}"
    )
    if setup.smoke:
        print(
            f"smoke: SYNTHETIC fixtures, smoke model settings, under {_show(setup.work_dir)}"
        )
    if dry_run:
        print("dry run: nothing is run or written")

    entry: dict = {}
    try:
        for stage in STAGES:
            if stage not in requested:
                log["stages"][stage] = {"status": "not requested"}
                continue
            stage_started = time.time()
            entry = log["stages"][stage] = {"status": "failed"}
            try:
                entry.update(HANDLERS[stage](setup, log, dry_run=dry_run, force=force))
            finally:
                entry["wall_seconds"] = round(time.time() - stage_started, 1)
            if dry_run and entry["status"] == "ran":
                entry["status"] = "planned"
        if metrics_only:
            entry = log["metrics_only"] = {"status": "failed"}
            entry.update(_prune_fold_tables(setup, log, dry_run))
    except BaseException as error:
        entry["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        log["pruned_tables"] = [
            p for p in log["pruned_tables"] if not resolve(p).exists()
        ]
        log["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        log["wall_seconds"] = round(time.time() - started, 1)
        if not dry_run:
            setup.record_path.parent.mkdir(parents=True, exist_ok=True)
            setup.record_path.write_text(json.dumps(log, indent=2, default=str) + "\n")
            print(f"wrote {_show(setup.record_path)} ({log['wall_seconds']} s)")
    return log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment", required=True, help="configs/experiments/<name>.yaml"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="stages to consider; they always run in the order above (default: all)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="override forecast.seeds")
    parser.add_argument(
        "--folds",
        nargs="*",
        type=int,
        help="override forecast.folds (no values: final runs only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun forecast runs whose tables exist, and re-evaluate",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="delete the fold tables after a successful evaluation",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; run and write nothing"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="whole chain on the synthetic fixtures: smoke settings, first seed and fold",
    )
    parser.add_argument(
        "--work-dir",
        help=f"--smoke outputs (default {SMOKE_WORK_DIR}/<name>); a new one runs every stage",
    )
    args = parser.parse_args(argv)
    if args.work_dir and not args.smoke:
        parser.error("--work-dir is only used with --smoke")
    if not resolve(args.experiment).exists():
        parser.error(f"{args.experiment} does not exist")

    experiment = load_yaml(args.experiment)
    if isinstance(experiment, dict) and isinstance(experiment.get("forecast"), dict):
        if args.seeds is not None:
            experiment["forecast"]["seeds"] = args.seeds
        if args.folds is not None:
            experiment["forecast"]["folds"] = args.folds
    try:
        run_experiment(
            experiment,
            stages=args.stages,
            force=args.force,
            dry_run=args.dry_run,
            smoke=args.smoke,
            metrics_only=args.metrics_only,
            work_dir=args.work_dir,
        )
    except ExperimentError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
