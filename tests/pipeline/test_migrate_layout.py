"""pipeline/migrate_layout.py: old-layout run directories move under the imputation run id, and the paths in
their run logs follow, including absolute paths written on another machine."""

import json

import pytest

from uncertainty_aware_planner.pipeline.migrate_layout import (
    main,
    migrate,
    rewrite_paths,
)

RUN_ID = "imp-none-censored"
NAME = "fc-tft-quantile-2025"
CLUSTER = "/scratch/user/uncertainty_aware_planner/data"  # another machine's checkout


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


@pytest.fixture
def root(tmp_path):
    """The old layout as pipeline/forecast.py wrote it, plus one directory that is already nested."""
    root = tmp_path / "data"
    for base, names in {
        "processed/forecast": ["final_calib.parquet", "final_test.parquet"],
        "processed/forecast_paper_split": ["final_test.parquet"],
    }.items():
        for name in names:
            path = root / base / NAME / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"table")
    write_json(
        root / "runs/forecasting" / NAME / "final/run_log.json",
        {
            "imputation_run_id": RUN_ID,
            "demand_daily": "data/processed/demand_daily/imp-none-censored.parquet",
            "model_config": {
                "output_dir": "data/processed/forecast",
                "smoke": {"output_dir": "data/processed/forecast_smoke"},
            },
            "checkpoint": f"{CLUSTER}/runs/forecasting/{NAME}/final/checkpoints/last.ckpt",
            "outputs": {
                "calib": {
                    "table": f"{CLUSTER}/processed/forecast/{NAME}/final_calib.parquet",
                    "n_rows": 42,
                }
            },
        },
    )
    write_json(
        root / "runs/forecasting_paper_split" / NAME / "final/run_log.json",
        {
            "imputation_run_id": RUN_ID,
            "outputs": {
                "test": {
                    "table": f"/home/me/repo/data/processed/forecast_paper_split/{NAME}/final_test.parquet"
                }
            },
        },
    )
    (root / "processed/forecast/imp-other/fc-tft-quantile-2025").mkdir(parents=True)
    return root


def test_moves_old_runs_and_rewrites_their_run_logs(root):
    migrate(root, RUN_ID)

    assert not (root / "processed/forecast" / NAME).exists()
    moved = root / "processed/forecast" / RUN_ID / NAME
    assert (moved / "final_test.parquet").read_bytes() == b"table"
    assert (root / "processed/forecast_paper_split" / RUN_ID / NAME).is_dir()
    assert (root / "processed/forecast/imp-other" / NAME).is_dir()  # already nested

    log = json.loads(
        (root / "runs/forecasting" / RUN_ID / NAME / "final/run_log.json").read_text()
    )
    assert log["checkpoint"] == (
        f"{CLUSTER}/runs/forecasting/{RUN_ID}/{NAME}/final/checkpoints/last.ckpt"
    )
    assert log["outputs"]["calib"] == {
        "table": f"{CLUSTER}/processed/forecast/{RUN_ID}/{NAME}/final_calib.parquet",
        "n_rows": 42,
    }
    assert (
        log["demand_daily"] == "data/processed/demand_daily/imp-none-censored.parquet"
    )
    assert log["model_config"] == {
        "output_dir": "data/processed/forecast",
        "smoke": {"output_dir": "data/processed/forecast_smoke"},
    }
    paper = json.loads(
        (
            root / "runs/forecasting_paper_split" / RUN_ID / NAME / "final/run_log.json"
        ).read_text()
    )
    assert paper["outputs"]["test"]["table"] == (
        f"/home/me/repo/data/processed/forecast_paper_split/{RUN_ID}/{NAME}/final_test.parquet"
    )

    [entry] = json.loads((root / "runs/forecasting/MIGRATED.json").read_text())
    assert entry["imputation_run_id"] == RUN_ID
    assert entry["moved"] == [{"from": NAME, "to": f"{RUN_ID}/{NAME}"}]
    assert entry["rewritten_run_logs"] == {f"{RUN_ID}/{NAME}/final/run_log.json": 2}
    assert (root / "processed/forecast/MIGRATED.json").exists()
    assert not (root / "runs/forecasting_smoke").exists()  # missing bases are skipped

    # Idempotent: nothing is left to move, so nothing changes, MIGRATED.json included.
    after = snapshot(root)
    assert migrate(root, RUN_ID) == {}
    assert snapshot(root) == after


def test_refuses_existing_destinations_and_other_imputation_runs(root, capsys):
    before = snapshot(root)
    with pytest.raises(ValueError, match=f"records imputation run {RUN_ID}"):
        migrate(root, "imp-dlinear-baseline")
    with pytest.raises(ValueError, match="not an imputation run id"):
        migrate(root, "fc-tft-quantile")
    assert snapshot(root) == before

    (root / "runs/forecasting" / RUN_ID / NAME).mkdir(parents=True)
    before = snapshot(root)
    with pytest.raises(FileExistsError, match="nothing moved"):
        migrate(root, RUN_ID)
    assert main(["--imputation-run-id", RUN_ID, "--root", str(root)]) == 1
    assert "destination exists" in capsys.readouterr().err
    assert snapshot(root) == before


def test_dry_run_changes_nothing(root, capsys):
    before = snapshot(root)
    argv = ["--imputation-run-id", RUN_ID, "--root", str(root), "--dry-run"]
    assert main(argv) == 0
    assert snapshot(root) == before
    out = capsys.readouterr().out
    assert "dry run: nothing is moved or written" in out
    assert f"runs/forecasting: {NAME} -> {RUN_ID}/{NAME}" in out
    assert f"2 paths in {NAME}/final/run_log.json" in out


def test_rewrites_whole_old_layout_segments_only():
    cases = {
        # relative, at the start of the string
        "processed/forecast/fc-a/fold1.parquet": "processed/forecast/imp-x/fc-a/fold1.parquet",
        # ends at the run directory
        "/x/data/runs/forecasting_smoke/fc-a": "/x/data/runs/forecasting_smoke/imp-x/fc-a",
        # already nested, a longer directory name, no run directory, not a path
        "/x/processed/forecast/imp-y/fc-a/final_test.parquet": None,
        "/x/myprocessed/forecast/fc-a/final_test.parquet": None,
        "data/processed/forecast": None,
        "fc-a": None,
    }
    for old, new in cases.items():
        assert rewrite_paths(old, "imp-x") == (new or old, int(new is not None))
    nested = {"a": ["runs/forecasting/fc-a/final", 3, None], "b": {"c": "fc-a"}}
    assert rewrite_paths(nested, "imp-x") == (
        {"a": ["runs/forecasting/imp-x/fc-a/final", 3, None], "b": {"c": "fc-a"}},
        1,
    )
