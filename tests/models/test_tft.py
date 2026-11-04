import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uncertainty_aware_planner.data.splits import final_windows, fold_windows
from uncertainty_aware_planner.models.forecasting.forecast_table import (
    COLUMNS,
    long_quantiles,
)
from uncertainty_aware_planner.pipeline.forecast import run

N_SERIES = 6
HORIZON = 7


def check_contract(table: pd.DataFrame, demand: pd.DataFrame):
    assert list(table.columns) == COLUMNS
    assert table["horizon"].dtype == "uint8"
    assert set(table["horizon"]) == set(range(1, HORIZON + 1))
    assert (
        table["target_date"]
        == table["origin_date"]
        + pd.to_timedelta(table["horizon"], unit="D").dt.to_pytimedelta()
    ).all()
    assert (table["date"] == table["target_date"]).all()
    assert isinstance(table["origin_date"].iloc[0], date)
    assert (table["q025"] >= 0).all()
    assert (table["q025"] <= table["estimated_demand"]).all()
    assert (table["estimated_demand"] <= table["q975"]).all()
    assert not table.duplicated(
        ["store_id", "product_id", "origin_date", "horizon"]
    ).any()
    assert table[["q025_cal", "q975_cal", "qhat95"]].isna().all().all()
    assert (table["forecast_run_id"] == "fc-tft-quantile-7").all()
    assert (table["imputation_run_id"] == "imp-fixture-0").all()

    actual = demand.assign(target_date=pd.to_datetime(demand["date"]).dt.date)[
        ["store_id", "product_id", "target_date", "demand_total"]
    ]
    joined = table.merge(actual, on=["store_id", "product_id", "target_date"])
    assert len(joined) == len(table)
    np.testing.assert_allclose(joined["actual_demand"], joined["demand_total"])


def test_fold_run_writes_contract_table(cfg, demand, tft_cfg):
    log = run(tft_cfg, cfg, demand, seed=7, fold=1)
    table = pd.read_parquet(log["outputs"]["val"]["table"])
    check_contract(table, demand)

    val = fold_windows(cfg["splits"])[1]["val"]
    assert (table["forecast_split"] == "val").all()
    assert (table["forecast_fold"] == 1).all()
    # rolling origins across the 28-day val window: 22 forecasts of 7 days per series
    assert len(table) == N_SERIES * (val.days - HORIZON + 1) * HORIZON
    assert table["target_date"].min() == val.start
    assert table["target_date"].max() == val.end

    run_dir = Path(tft_cfg["run_dir"]) / "fc-tft-quantile-7" / "fold1"
    saved = json.loads((run_dir / "run_log.json").read_text())
    assert saved["seed"] == 7 and saved["fold"] == 1
    assert Path(saved["checkpoint"]).exists()
    assert {"q025", "q500", "q975"} <= set(
        pd.read_parquet(run_dir / "quantiles_val.parquet").columns
    )


def test_final_run_forecasts_calib_and_test_once(cfg, demand, tft_cfg):
    log = run(tft_cfg, cfg, demand, seed=7, eval_parts=["calib", "test"])
    final = final_windows(cfg["splits"])
    for part in ("calib", "test"):
        table = pd.read_parquet(log["outputs"][part]["table"])
        check_contract(table, demand)
        assert len(table) == N_SERIES * HORIZON
        assert table["forecast_fold"].isna().all()
        assert (table["forecast_split"] == part).all()
        assert (table["origin_date"] == final[part].start - timedelta(days=1)).all()
    # test-week targets are hidden from the model but still reported for evaluation (null on stockout days,
    # which no imputation run covers)
    assert table["actual_demand"].notna().any()


def test_quantiles_are_sorted_and_clipped():
    q = np.array(
        [[[0.5, -1.0, 2.0], [3.0, 1.0, 2.0]]]
    )  # one sample, two days, levels unordered
    index = pd.DataFrame({"time_idx": [10], "store_id": ["4"], "product_id": ["2"]})
    rows, stats = long_quantiles(
        q, index, np.array([2]), [0.5, 0.025, 0.975], date(2024, 3, 28)
    )
    assert list(rows[["q025", "q500", "q975"]].iloc[0]) == [0.0, 0.5, 2.0]
    assert list(rows[["q025", "q500", "q975"]].iloc[1]) == [1.0, 2.0, 3.0]
    assert stats["quantile_crossing_rate"] == 0.5 and stats[
        "negative_clip_rate"
    ] == pytest.approx(1 / 6)
    assert rows["origin_date"].iloc[0] == date(2024, 4, 6)
    assert list(rows["target_date"]) == [date(2024, 4, 7), date(2024, 4, 8)]
