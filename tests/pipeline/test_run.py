"""pipeline/run.py: the stage plan, skip logic, both protocols and --metrics-only, with forecast.run and the
evaluation CLI replaced by fakes that follow their documented interfaces; then the real chain on the fixtures."""

import argparse
import glob
import importlib
import json
import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fixtures.synthetic_raw_series import fixture_config
from uncertainty_aware_planner.data.config import REPO_ROOT
from uncertainty_aware_planner.pipeline.run import (
    EVALUATE_MODULE,
    FORECAST_MODULE,
    STAGES,
    ExperimentError,
    main,
    run_experiment,
)

RUN_ID = "imp-none-censored"
MODEL_ID = "fc-fake-point"
FORECAST_STAGES = ["forecast", "evaluate"]


class FakeForecast:
    """pipeline.forecast.run stand-in: empty tables, manifests and run logs in the documented layout."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        model_cfg,
        data_cfg,
        demand_daily,
        seed,
        fold=None,
        eval_parts=None,
        demand_daily_path=None,
    ):
        imputation_run_id = str(demand_daily["imputation_run_id"].iloc[0])
        rid = f"{model_cfg['model_id']}-{seed}"
        parts = ["val"] if fold is not None else list(eval_parts)
        self.calls.append(
            {
                "seed": seed,
                "fold": fold,
                "parts": parts,
                "output_dir": model_cfg["output_dir"],
                "run_dir": model_cfg["run_dir"],
                "config_path": model_cfg["config_path"],
                "train": list(data_cfg["splits"]["final"]["train"]),
                "demand_daily_path": demand_daily_path,
            }
        )
        table_dir = Path(model_cfg["output_dir"]) / imputation_run_id / rid
        run_dir = Path(model_cfg["run_dir"]) / imputation_run_id / rid
        run_dir = run_dir / (f"fold{fold}" if fold is not None else "final")
        table_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for part in parts:
            name = f"fold{fold}" if fold is not None else f"final_{part}"
            table = table_dir / f"{name}.parquet"
            table.write_bytes(b"table")
            table.with_name(table.name + ".manifest.json").write_text("{}")
            outputs[part] = {"table": str(table)}
        (run_dir / "run_log.json").write_text(json.dumps({"outputs": outputs}))
        return {"run_id": rid, "outputs": outputs}


class FakeEvaluation:
    """evaluate_forecasting.main stand-in: parses the documented arguments, writes metrics.csv."""

    def __init__(self):
        self.calls = []
        self.exit_code = 0

    def __call__(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--tables", nargs="+", required=True)
        for flag in ("--truth", "--slices", "--quantiles-root", "--out"):
            parser.add_argument(flag, required=True)
        parser.add_argument(
            "--point", choices=["estimated_demand", "q500"], required=True
        )
        parser.add_argument("--protocol", choices=["project", "paper"], required=True)
        parser.add_argument("--no-ledger", action="store_true")
        args = parser.parse_args(argv)
        tables = sorted(t for pattern in args.tables for t in glob.glob(pattern))
        self.calls.append({"args": args, "tables": tables})
        if self.exit_code == 0:
            out = Path(args.out) / RUN_ID / args.protocol
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.csv").write_text("metric,value\n")
        return self.exit_code


@pytest.fixture
def forecast_runs(monkeypatch):
    fake = FakeForecast()
    monkeypatch.setattr(importlib.import_module(FORECAST_MODULE), "run", fake)
    return fake


@pytest.fixture
def evaluation(monkeypatch):
    fake = FakeEvaluation()
    module = types.ModuleType(EVALUATE_MODULE)
    module.main = fake
    monkeypatch.setitem(sys.modules, EVALUATE_MODULE, module)
    return fake


@pytest.fixture
def experiment(tmp_path):
    """Data config, model config and every output under tmp_path; demand_daily already exists."""
    data_cfg = {
        k: v for k, v in fixture_config(tmp_path / "data").items() if k != "splits"
    }
    data_cfg["splits_config"] = str(REPO_ROOT / "configs" / "splits.yaml")
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump(data_cfg))
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump(
            {
                "architecture": "fake",
                "model_id": MODEL_ID,
                "data_config": str(data_config),
                "output_dir": str(tmp_path / "forecast"),
                "run_dir": str(tmp_path / "runs"),
            }
        )
    )
    demand = tmp_path / "data" / "processed" / "demand_daily" / f"{RUN_ID}.parquet"
    demand.parent.mkdir(parents=True)
    pd.DataFrame({"imputation_run_id": [RUN_ID, RUN_ID]}).to_parquet(demand)
    return {
        "name": "fake_censored",
        "data_config": str(data_config),
        "imputation": {"run_id": RUN_ID, "source": "censored"},
        "forecast": {
            "config": str(model_config),
            "seeds": [1, 2],
            "folds": [1, 2],
            "final_parts": ["calib", "test"],
            "protocol": "project",
        },
        "evaluate": {
            "point": "estimated_demand",
            "truth": str(demand),
            "slices": str(tmp_path / "slices.parquet"),
            "out": str(tmp_path / "results"),
        },
    }


def record(tmp_path, protocol="project", name="fake_censored"):
    path = tmp_path / "results" / RUN_ID / protocol / f"experiment_{name}.json"
    return json.loads(path.read_text())


def test_dry_run_prints_the_plan_and_runs_nothing(
    experiment, tmp_path, forecast_runs, evaluation, capsys
):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    before = sorted(tmp_path.rglob("*"))
    argv = ["--experiment", str(path), "--dry-run", "--seeds", "7", "--folds", "2"]
    assert main(argv) == 0

    assert sorted(tmp_path.rglob("*")) == before
    assert forecast_runs.calls == [] and evaluation.calls == []
    out = capsys.readouterr().out
    assert "dry run: nothing is run or written" in out
    for line in (
        "run data: local parquet source",
        f"run impute: {tmp_path}/data/processed/imputed_hourly/{RUN_ID}.parquet missing",
        "skip daily:",
        "run forecast: 2 of 2 runs",
        "  run seed 7 fold 2:",
        "  run seed 7 final:",
        "run evaluate: forecast runs planned",
    ):
        assert line in out
    assert "seed 1" not in out and "fold 1" not in out


def test_runs_folds_then_the_final_run_per_seed_and_evaluates(
    experiment, tmp_path, forecast_runs, evaluation
):
    run_experiment(experiment, stages=FORECAST_STAGES)

    assert [(c["seed"], c["fold"], c["parts"]) for c in forecast_runs.calls] == [
        (1, 1, ["val"]),
        (1, 2, ["val"]),
        (1, None, ["calib", "test"]),
        (2, 1, ["val"]),
        (2, 2, ["val"]),
        (2, None, ["calib", "test"]),
    ]
    first = forecast_runs.calls[0]
    assert first["output_dir"] == str(tmp_path / "forecast")
    assert first["run_dir"] == str(tmp_path / "runs")
    assert first["config_path"] == experiment["forecast"]["config"]
    assert first["train"] == [date(2024, 3, 28), date(2024, 6, 18)]

    [call] = evaluation.calls
    args = call["args"]
    assert args.tables == [
        str(tmp_path / "forecast" / RUN_ID / f"{MODEL_ID}-{seed}" / "*.parquet")
        for seed in (1, 2)
    ]
    assert (
        len(call["tables"]) == 2 * 4
    )  # fold1, fold2, final_calib, final_test per seed
    assert (args.point, args.protocol, args.no_ledger) == (
        "estimated_demand",
        "project",
        False,
    )
    assert args.quantiles_root == str(tmp_path / "runs")
    assert args.truth == experiment["evaluate"]["truth"]
    assert args.slices == experiment["evaluate"]["slices"]
    assert args.out == str(tmp_path / "results")

    saved = record(tmp_path)
    assert saved["experiment"]["forecast"]["seeds"] == [1, 2]
    assert {stage: entry["status"] for stage, entry in saved["stages"].items()} == {
        "data": "not requested",
        "impute": "not requested",
        "daily": "not requested",
        "forecast": "ran",
        "evaluate": "ran",
    }
    assert "wall_seconds" in saved["stages"]["forecast"]
    assert [r["status"] for r in saved["runs"]] == ["ran"] * 6
    assert len(saved["tables"]) == 8 and saved["git_commit"]


def test_paper_protocol_trains_through_calib_and_forecasts_the_test_week(
    experiment, tmp_path, forecast_runs, evaluation
):
    experiment["forecast"]["protocol"] = "paper"
    run_experiment(experiment, stages=FORECAST_STAGES)

    assert [(c["seed"], c["fold"], c["parts"]) for c in forecast_runs.calls] == [
        (1, None, ["test"]),
        (2, None, ["test"]),
    ]
    first = forecast_runs.calls[0]
    assert first["output_dir"] == str(tmp_path / "forecast_paper_split")
    assert first["run_dir"] == str(tmp_path / "runs_paper_split")
    assert first["train"] == [date(2024, 3, 28), date(2024, 6, 25)]
    table_dir = tmp_path / "forecast_paper_split" / RUN_ID / f"{MODEL_ID}-1"
    assert sorted(p.name for p in table_dir.glob("*.parquet")) == ["final_test.parquet"]

    args = evaluation.calls[0]["args"]
    assert args.protocol == "paper"
    assert args.quantiles_root == str(tmp_path / "runs_paper_split")
    assert record(tmp_path, "paper")["stages"]["evaluate"]["status"] == "ran"


def test_existing_tables_are_skipped_and_force_reruns(
    experiment, tmp_path, forecast_runs, evaluation, capsys
):
    run_experiment(experiment, stages=FORECAST_STAGES)
    assert (len(forecast_runs.calls), len(evaluation.calls)) == (6, 1)

    capsys.readouterr()
    run_experiment(experiment, stages=FORECAST_STAGES)
    out = capsys.readouterr().out
    assert (len(forecast_runs.calls), len(evaluation.calls)) == (6, 1)
    assert "skip forecast: all 6 runs done" in out
    assert "skip evaluate:" in out

    # A table without its manifest does not count as written.
    table = tmp_path / "forecast" / RUN_ID / f"{MODEL_ID}-2" / "fold1.parquet"
    table.with_name(table.name + ".manifest.json").unlink()
    run_experiment(experiment, stages=FORECAST_STAGES)
    assert [(c["seed"], c["fold"]) for c in forecast_runs.calls[6:]] == [(2, 1)]
    assert len(evaluation.calls) == 2

    run_experiment(experiment, stages=FORECAST_STAGES, force=True)
    assert (len(forecast_runs.calls), len(evaluation.calls)) == (7 + 6, 3)


def test_metrics_only_deletes_fold_tables_after_the_evaluation(
    experiment, tmp_path, forecast_runs, evaluation
):
    log = run_experiment(experiment, stages=FORECAST_STAGES, metrics_only=True)

    [call] = evaluation.calls
    assert sum(Path(t).name.startswith("fold") for t in call["tables"]) == 4
    for seed in (1, 2):
        table_dir = tmp_path / "forecast" / RUN_ID / f"{MODEL_ID}-{seed}"
        assert sorted(p.name for p in table_dir.iterdir()) == [
            "final_calib.parquet",
            "final_calib.parquet.manifest.json",
            "final_test.parquet",
            "final_test.parquet.manifest.json",
        ]
        run_dir = tmp_path / "runs" / RUN_ID / f"{MODEL_ID}-{seed}"
        assert (run_dir / "fold1" / "run_log.json").exists()
    assert len(log["pruned_tables"]) == 4
    assert (tmp_path / "results" / RUN_ID / "project" / "metrics.csv").exists()

    # Later invocations count the deleted fold tables as done instead of retraining.
    run_experiment(experiment, stages=FORECAST_STAGES)
    assert (len(forecast_runs.calls), len(evaluation.calls)) == (6, 1)
    assert len(record(tmp_path)["pruned_tables"]) == 4


def test_metrics_only_keeps_fold_tables_when_the_evaluation_fails(
    experiment, tmp_path, forecast_runs, evaluation
):
    evaluation.exit_code = 1
    with pytest.raises(RuntimeError, match="exited with status 1"):
        run_experiment(experiment, stages=FORECAST_STAGES, metrics_only=True)
    assert len(list((tmp_path / "forecast").rglob("fold*.parquet"))) == 4
    saved = record(tmp_path)
    assert saved["stages"]["evaluate"]["status"] == "failed"
    assert "exited with status 1" in saved["stages"]["evaluate"]["error"]


def test_imputed_source_needs_the_impute_output(experiment, tmp_path, capsys):
    experiment["imputation"] = {"run_id": "imp-dlinear-baseline", "source": "imputed"}
    with pytest.raises(ExperimentError, match="pipeline/impute.py"):
        run_experiment(experiment, stages=["impute"])

    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    assert main(["--experiment", str(path), "--stages", "impute"]) == 2
    assert "pipeline/impute.py" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("imputation", "run_id", "imp-other", "source censored is imputation run"),
        ("forecast", "protocol", "rolling", "forecast.protocol must be one of"),
        ("forecast", "final_parts", ["val"], "forecast.final_parts must be a subset"),
        ("evaluate", "point", "q975", "evaluate.point must be one of"),
    ],
)
def test_malformed_experiments_are_refused(experiment, section, key, value, message):
    experiment[section][key] = value
    with pytest.raises(ExperimentError, match=message):
        run_experiment(experiment, dry_run=True)


def test_smoke_chain_runs_the_real_stages(tmp_path, evaluation, capsys):
    """load, censored, daily and the real forecaster on the fixtures; the evaluation CLI is the stand-in."""
    argv = [
        "--experiment",
        "configs/experiments/dlinear_baseline_censored.yaml",
        "--smoke",
        "--work-dir",
        str(tmp_path),
    ]
    assert main(argv) == 0
    saved = json.loads(
        (
            tmp_path
            / "results"
            / "forecasting"
            / RUN_ID
            / "project"
            / "experiment_dlinear_baseline_censored.json"
        ).read_text()
    )
    assert {s: e["status"] for s, e in saved["stages"].items()} == dict.fromkeys(
        STAGES, "ran"
    )
    assert [(r["seed"], r["fold"]) for r in saved["runs"]] == [(2025, 1), (2025, None)]
    table_dir = (
        tmp_path / "processed" / "forecast_smoke" / RUN_ID / "fc-dlinear-baseline-2025"
    )
    assert sorted(p.name for p in table_dir.glob("*.parquet")) == [
        "final_calib.parquet",
        "final_test.parquet",
        "fold1.parquet",
    ]
    args = evaluation.calls[0]["args"]
    processed = tmp_path / "processed"
    assert args.truth == str(processed / "demand_daily" / f"{RUN_ID}.parquet")
    assert args.slices == str(processed / "slices.parquet")
    assert args.quantiles_root == str(tmp_path / "runs" / "forecasting_smoke")
    assert args.no_ledger

    capsys.readouterr()
    assert main(argv) == 0
    out = capsys.readouterr().out
    for stage in ("impute", "daily", "forecast", "evaluate"):
        assert f"skip {stage}:" in out
    assert len(evaluation.calls) == 1


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not (
        _importable("uncertainty_aware_planner.models.forecasting.dlinear")
        and _importable(EVALUATE_MODULE)
    ),
    reason="the DLinear forecaster or pipeline.evaluate_forecasting is not importable yet",
)
def test_smoke_runs_the_whole_chain(tmp_path):
    argv = [
        "--experiment",
        "configs/experiments/dlinear_baseline_censored.yaml",
        "--smoke",
        "--work-dir",
        str(tmp_path),
    ]
    assert main(argv) == 0
    out = tmp_path / "results" / "forecasting" / RUN_ID / "project"
    saved = json.loads((out / "experiment_dlinear_baseline_censored.json").read_text())
    assert {s: e["status"] for s, e in saved["stages"].items()} == dict.fromkeys(
        STAGES, "ran"
    )
    assert (out / "metrics.csv").exists()
