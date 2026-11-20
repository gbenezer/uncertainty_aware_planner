"""pipeline/impute.py on the SYNTHETIC fixtures with tiny models: contract columns and invariants, the
demand_daily handoff, the masked evaluation rows, reproducibility and no overwrites."""

import numpy as np
import pandas as pd
import pytest

from fixtures.synthetic_raw_series import fixture_config
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.daily import build_demand_daily
from uncertainty_aware_planner.data.masks import make_eval_masks
from uncertainty_aware_planner.data.outputs import write_parquet
from uncertainty_aware_planner.data.splits import final_windows, fold_windows
from uncertainty_aware_planner.pipeline import impute

TINY = {
    "n_steps": 672,
    "n_features": 9,
    "batch_size": 8,
    "epochs": 1,
    "patience": None,
    "optimizer_parameters": None,
}
DLINEAR = {
    **TINY,
    "moving_avg_window_size": 3,
    "individual": False,
    "d_model": 8,
    "ORT_weight": 1,
    "MIT_weight": 1,
}
BASELINE = {"name": "DLinear_Imputation_Baseline", **DLINEAR}
QUANTILE = {
    "name": "DLinear_Imputation_Quantile",
    **DLINEAR,
    "quantiles": [0.025, 0.975],
}
TIMESNET = {
    "name": "TimesNet_Imputation_Quantile",
    **TINY,
    "n_layers": 1,
    "top_k": 1,
    "d_model": 8,
    "d_ffn": 8,
    "n_kernels": 1,
    "dropout": 0.0,
    "apply_nonstationary_norm": False,
    "quantiles": [0.025, 0.5, 0.975],
}


def workspace(path):
    """Fixture data config writing under `path`, with the shared eval masks in place."""
    cfg = fixture_config(path)
    cube = load_cube(cfg)
    masks = make_eval_masks(cube, cfg["splits"])
    write_parquet(masks, path / "processed" / "eval_masks.parquet")
    return cfg, cube, masks


def check_contract(table, quantile: bool):
    assert list(table.columns) == impute.COLUMNS
    assert table["estimated_demand"].notna().all()
    assert (table["estimated_demand"] >= 0).all()
    assert table[["q025_cal", "q975_cal", "qhat95"]].isna().all().all()
    if quantile:
        assert (table["q025"] <= table["estimated_demand"]).all()
        assert (table["estimated_demand"] <= table["q975"]).all()
        assert (table["q025"] >= 0).all()
    else:
        assert table[["q025", "q975"]].isna().all().all()


def test_model_id():
    assert impute.model_id(BASELINE) == "imp-dlinear-baseline"
    assert impute.model_id(QUANTILE) == "imp-dlinear-quantile"
    assert impute.model_id(TIMESNET) == "imp-timesnet-quantile-median"
    with pytest.raises(ValueError, match="unknown imputation model"):
        impute.model_id({"name": "ImputeFormer_Imputation_Quantile"})


def test_quantile_config_needs_the_interval_levels(tmp_path):
    cfg, _, _ = workspace(tmp_path)
    with pytest.raises(ValueError, match="needs quantiles"):
        impute.run({**QUANTILE, "quantiles": [0.1, 0.9]}, cfg, seed=7, fold=1)


def test_final_run_fills_every_stockout_and_feeds_demand_daily(tmp_path):
    cfg, cube, masks = workspace(tmp_path)
    log = impute.run(QUANTILE, cfg, seed=7, run_root=tmp_path / "runs")
    rid = "imp-dlinear-quantile-7"
    assert log["run_id"] == rid
    assert (tmp_path / "runs" / rid / "final" / "run_log.json").exists()

    recovery = pd.read_parquet(
        tmp_path / "processed" / "imputed_hourly" / f"{rid}.parquet"
    )
    check_contract(recovery, quantile=True)
    assert recovery["actual_demand"].isna().all()
    assert recovery["impute_fold"].isna().all()
    assert set(recovery["impute_split"]) == {"train", "calib"}
    final = final_windows(cfg["splits"])
    a, b = cube.day_index(final["train"].start), cube.day_index(final["calib"].end) + 1
    assert len(recovery) == cube.stockout[:, a:b].sum()

    demand, _ = build_demand_daily(cube, recovery, rid, cfg["splits"])
    through_calib = pd.to_datetime(demand["date"]) <= pd.Timestamp(final["calib"].end)
    assert demand.loc[through_calib, "demand_recovered"].notna().all()

    calib = pd.read_parquet(
        tmp_path / "processed" / "imputation" / rid / "final_calib.parquet"
    )
    check_contract(calib, quantile=True)
    assert set(calib["impute_split"]) == {"calib"}
    assert calib["actual_demand"].notna().sum() == (masks["split"] == "calib").sum()


@pytest.mark.parametrize(
    "model_cfg", [BASELINE, QUANTILE, TIMESNET], ids=lambda c: c["name"]
)
def test_fold_run_scores_the_fold_masks(tmp_path, model_cfg):
    cfg, _, masks = workspace(tmp_path)
    log = impute.run(model_cfg, cfg, seed=7, fold=1, run_root=tmp_path / "runs")
    table = pd.read_parquet(log["outputs"]["val"]["table"])
    check_contract(table, quantile=model_cfg["name"] != impute.BASELINE)
    assert set(table["impute_split"]) == {"val"}
    assert (table["impute_fold"] == 1).all()
    val = fold_windows(cfg["splits"])[1]["val"]
    days = pd.to_datetime(table["date"])
    assert days.between(pd.Timestamp(val.start), pd.Timestamp(val.end)).all()
    fold_masks = masks[(masks["split"] == "val") & (masks["fold"] == 1)]
    assert table["actual_demand"].notna().sum() == len(fold_masks)


def test_same_seed_gives_the_same_table(tmp_path):
    tables = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        cfg, _, _ = workspace(tmp_path / name)
        log = impute.run(
            QUANTILE, cfg, seed=7, fold=1, run_root=tmp_path / name / "runs"
        )
        tables.append(pd.read_parquet(log["outputs"]["val"]["table"]))
    pd.testing.assert_frame_equal(*tables)


def test_never_overwrites(tmp_path):
    cfg, _, _ = workspace(tmp_path)
    impute.run(BASELINE, cfg, seed=7, fold=1, run_root=tmp_path / "runs")
    with pytest.raises(FileExistsError):
        impute.run(BASELINE, cfg, seed=7, fold=1, run_root=tmp_path / "runs")


def test_midpoint_is_the_estimate_for_quantile_models(tmp_path):
    cfg, _, _ = workspace(tmp_path)
    log = impute.run(TIMESNET, cfg, seed=7, fold=2, run_root=tmp_path / "runs")
    table = pd.read_parquet(log["outputs"]["val"]["table"])
    np.testing.assert_allclose(
        table["estimated_demand"], (table["q025"] + table["q975"]) / 2
    )
