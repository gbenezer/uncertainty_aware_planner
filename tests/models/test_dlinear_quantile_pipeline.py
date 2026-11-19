import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.splits import final_windows, fold_windows
from uncertainty_aware_planner.models.forecasting.forecast_table import COLUMNS
from uncertainty_aware_planner.pipeline.forecast import run

HORIZON = 7
N_SERIES = 6
MODEL_ID = "fc-dlinear-quantile-test-7"


def check_contract(table):
    """Check the DLinear forecast output format."""
    assert list(table.columns) == COLUMNS
    assert set(table["horizon"]) == set(range(1, HORIZON + 1))
    assert (table["q025"] >= 0).all()
    assert (table["q025"] <= table["estimated_demand"]).all()
    assert (table["estimated_demand"] <= table["q975"]).all()
    assert (table["forecast_run_id"] == MODEL_ID).all()
    assert (table["imputation_run_id"] == "imp-fixture-0").all()
    assert table[["q025_cal", "q975_cal", "qhat95"]].isna().all().all()


def test_dlinear_fold(cfg, demand, dlinear_quantile_cfg):
    """Check validation-fold forecasts and saved quantiles."""
    log = run(dlinear_quantile_cfg, cfg, demand, seed=7, fold=1)

    table = pd.read_parquet(log["outputs"]["val"]["table"])
    check_contract(table)

    val = fold_windows(cfg["splits"])[1]["val"]
    assert (table["forecast_split"] == "val").all()
    assert (table["forecast_fold"] == 1).all()
    assert len(table) == N_SERIES * (val.days - HORIZON + 1) * HORIZON

    run_dir = (
        Path(dlinear_quantile_cfg["run_dir"]) / "imp-fixture-0" / MODEL_ID / "fold1"
    )

    saved = json.loads((run_dir / "run_log.json").read_text())
    assert saved["seed"] == 7
    assert Path(saved["checkpoint"]).exists()

    quantiles = pd.read_parquet(run_dir / "quantiles_val.parquet")
    merged = table.merge(
        quantiles,
        on=["store_id", "product_id", "origin_date", "horizon"],
    )

    assert len(merged) == len(table)
    np.testing.assert_allclose(merged["estimated_demand"], merged["q500"])


def test_dlinear_final(cfg, demand, dlinear_quantile_cfg):
    """Check final calibration and test forecasts."""
    log = run(
        dlinear_quantile_cfg,
        cfg,
        demand,
        seed=7,
        eval_parts=["calib", "test"],
    )

    final = final_windows(cfg["splits"])

    for part in ("calib", "test"):
        table = pd.read_parquet(log["outputs"][part]["table"])
        check_contract(table)

        assert len(table) == N_SERIES * HORIZON
        assert (table["forecast_split"] == part).all()
        assert table["forecast_fold"].isna().all()
        assert (table["origin_date"] == final[part].start - timedelta(days=1)).all()
