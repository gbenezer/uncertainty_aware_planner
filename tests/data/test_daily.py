from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.data.helpers import fake_imputed_hourly
from uncertainty_aware_planner.data.daily import (
    COLUMNS,
    apply_edit_log,
    build_demand_daily,
)

TRAIN_START, CALIB_END = date(2024, 3, 28), date(2024, 6, 25)


@pytest.fixture(scope="module")
def demand(cube, cfg, masks):
    imputed = fake_imputed_hourly(cube, TRAIN_START, CALIB_END, value=0.5, masks=masks)
    daily, _ = build_demand_daily(cube, imputed, "imp-test-run-0", cfg["splits"])
    return daily


def test_columns_follow_the_contract(demand):
    assert list(demand.columns) == COLUMNS
    assert demand["forecast_fold"].isna().all()
    assert (demand["imputation_run_id"] == "imp-test-run-0").all()


def test_observed_recovered_and_total(cube, demand):
    observed = np.where(cube.stockout, 0.0, cube.sales).sum(axis=-1).reshape(-1)
    assert np.allclose(demand["demand_observed"], observed)
    n_stockout = cube.stockout.sum(axis=-1).reshape(-1)
    in_range = (demand["date"] <= pd.Timestamp(CALIB_END)).to_numpy()
    assert np.allclose(demand["demand_recovered"][in_range], 0.5 * n_stockout[in_range])
    assert np.allclose(
        demand["demand_total"][in_range],
        observed[in_range] + 0.5 * n_stockout[in_range],
    )


def test_days_without_imputation_are_null_not_zero(cube, demand):
    test_week = (demand["date"] > pd.Timestamp(CALIB_END)).to_numpy()
    n_stockout = cube.stockout.sum(axis=-1).reshape(-1)
    assert demand["demand_recovered"][test_week & (n_stockout > 0)].isna().all()
    assert (demand["demand_recovered"][test_week & (n_stockout == 0)] == 0).all()


def test_masked_evaluation_rows_never_feed_demand(cube, cfg, masks):
    # The masked rows in the fixture carry estimated_demand 99; they must be ignored.
    with_masks = fake_imputed_hourly(cube, TRAIN_START, CALIB_END, masks=masks)
    without = fake_imputed_hourly(cube, TRAIN_START, CALIB_END)
    a, _ = build_demand_daily(cube, with_masks, "a", cfg["splits"])
    b, _ = build_demand_daily(cube, without, "a", cfg["splits"])
    pd.testing.assert_frame_equal(a, b)


def test_forecast_split_labels(demand):
    counts = demand.groupby("forecast_split").size() / 6
    assert counts.to_dict() == {"calib": 7, "test": 7, "train": 83}


def test_inconsistent_imputation_is_rejected(cube, cfg):
    imputed = fake_imputed_hourly(cube, TRAIN_START, CALIB_END)
    with pytest.raises(ValueError, match="part of their stockout hours"):
        build_demand_daily(cube, imputed.iloc[1:], "x", cfg["splits"])
    in_stock = np.argwhere(~cube.stockout[0, 0])[0, 0]
    extra = imputed.iloc[:1].assign(
        store_id=0,
        product_id=100,
        datetime=pd.Timestamp(cube.dates[0]) + pd.Timedelta(hours=int(in_stock)),
    )
    with pytest.raises(ValueError, match="not stockout hours"):
        build_demand_daily(cube, pd.concat([imputed, extra]), "x", cfg["splits"])
    with pytest.raises(ValueError, match=">= 0"):
        build_demand_daily(
            cube, imputed.assign(estimated_demand=-1.0), "x", cfg["splits"]
        )


def _edits(*rows):
    return pd.DataFrame(
        rows,
        columns=[
            "edit_id",
            "store_id",
            "product_id",
            "date",
            "field",
            "old_value",
            "new_value",
            "edited_at",
            "reason",
        ],
    )


def test_edit_overlay_newest_edit_wins_and_input_is_untouched(demand):
    before = demand.copy()
    edits = _edits(
        (
            "e1",
            0,
            100,
            "2024-04-01",
            "demand",
            1.0,
            5.0,
            pd.Timestamp("2026-11-01T10:00Z"),
            "store confirmed",
        ),
        (
            "e2",
            0,
            100,
            "2024-04-01",
            "demand",
            5.0,
            7.0,
            pd.Timestamp("2026-11-02T10:00Z"),
            "corrected again",
        ),
        (
            "e3",
            1,
            101,
            "2024-04-02",
            "discount",
            1.0,
            0.8,
            pd.Timestamp("2026-11-01T11:00Z"),
            "promo missing",
        ),
    )
    edited, applied = apply_edit_log(demand, edits)
    pd.testing.assert_frame_equal(demand, before)
    assert sorted(applied) == ["e2", "e3"]
    row = edited[
        (edited["store_id"] == 0)
        & (edited["product_id"] == 100)
        & (edited["date"] == "2024-04-01")
    ]
    assert row["demand_total"].item() == 7.0
    row = edited[
        (edited["store_id"] == 1)
        & (edited["product_id"] == 101)
        & (edited["date"] == "2024-04-02")
    ]
    assert row["discount"].item() == 0.8
    untouched = edited.drop(index=row.index).drop(columns="demand_total")
    pd.testing.assert_frame_equal(
        untouched, demand.drop(index=row.index).drop(columns="demand_total")
    )


def test_edit_overlay_rejects_bad_edits(demand):
    stamp = pd.Timestamp("2026-11-01T10:00Z")
    with pytest.raises(ValueError, match="reason"):
        apply_edit_log(
            demand, _edits(("e1", 0, 100, "2024-04-01", "demand", 1.0, 5.0, stamp, " "))
        )
    with pytest.raises(ValueError, match="cannot be edited"):
        apply_edit_log(
            demand,
            _edits(("e1", 0, 100, "2024-04-01", "product_id", 1.0, 5.0, stamp, "x")),
        )
    with pytest.raises(ValueError, match="do not exist"):
        apply_edit_log(
            demand, _edits(("e1", 9, 100, "2024-04-01", "demand", 1.0, 5.0, stamp, "x"))
        )
