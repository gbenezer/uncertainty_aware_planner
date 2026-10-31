from datetime import date, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from fixtures.synthetic_raw_series import FIXTURE_DIR, make_raw_series
from uncertainty_aware_planner.data.clean import DataValidationError
from uncertainty_aware_planner.data.cube import build_cube


def test_committed_fixture_files_match_the_generator(raw_tables):
    train, test = raw_tables
    assert pq.read_table(FIXTURE_DIR / "train.parquet").equals(train)
    assert pq.read_table(FIXTURE_DIR / "eval.parquet").equals(test)


def test_stockout_flag_polarity(cube, raw_tables):
    # 1 in hours_stock_status means OUT of stock (EDA Q1); the cube's stockout must be True exactly there.
    for table in raw_tables:
        series = cube.series_index(
            table["store_id"].to_numpy(), table["product_id"].to_numpy()
        )
        days = (
            pc.cast(table["dt"], pa.date32()).to_numpy().astype("datetime64[D]")
            - cube.dates[0]
        ).astype(int)
        flags = np.array(table["hours_stock_status"].to_pylist())
        assert (cube.stockout[series, days] == (flags == 1)).all()
    assert cube.stockout.any() and not cube.stockout.all()


def test_hourly_sales_add_up_to_daily(cube):
    daily = cube.daily["sale_amount"].to_numpy().reshape(cube.n_series, cube.n_days)
    assert np.allclose(cube.sales.sum(axis=-1), daily)


def test_discount_cleaning(cube):
    assert cube.cleaning_report == {
        "train": {"discount_missing_code": 3, "discount_above_max": 1},
        "eval": {"discount_missing_code": 1, "discount_above_max": 0},
    }
    assert cube.daily["discount"].min() > 0
    assert cube.daily["discount"].max() <= 1.0
    assert set(np.round(cube.daily["discount"].unique(), 6)) == {0.8, 1.0}


def test_rows_are_sorted_and_series_complete(cube):
    keys = (
        cube.daily[["store_id", "product_id"]]
        .to_numpy()
        .reshape(cube.n_series, cube.n_days, 2)
    )
    assert (keys == keys[:, :1]).all()
    assert (np.diff(cube.dates) == np.timedelta64(1, "D")).all()
    assert cube.n_days == 97


def _fail(cfg, train, test):
    with pytest.raises(DataValidationError) as error:
        build_cube(train, test, cfg)
    return " ".join(error.value.failures)


def test_validation_catches_bad_lists_and_flags(cfg):
    train, test = make_raw_series()
    short = train.set_column(
        train.column_names.index("hours_sale"),
        "hours_sale",
        pa.array(
            [v[:23] for v in train["hours_sale"].to_pylist()], pa.list_(pa.float64())
        ),
    )
    assert "24 values" in _fail(cfg, short, test)

    flags = [[2] + v[1:] for v in train["hours_stock_status"].to_pylist()]
    bad = train.set_column(
        train.column_names.index("hours_stock_status"),
        "hours_stock_status",
        pa.array(flags, pa.list_(pa.int64())),
    )
    assert "other than 0 and 1" in _fail(cfg, bad, test)


def test_validation_catches_duplicates_and_sum_mismatch(cfg):
    train, test = make_raw_series()
    duplicated = pa.concat_tables(
        [train.slice(0, train.num_rows - 1), train.slice(0, 1)]
    )
    assert "duplicate" in _fail(cfg, duplicated, test)

    amounts = train["sale_amount"].to_numpy().copy()
    amounts[0] += 1.0
    wrong = train.set_column(
        train.column_names.index("sale_amount"), "sale_amount", pa.array(amounts)
    )
    assert "differ from sale_amount" in _fail(cfg, wrong, test)


def test_validation_catches_calendar_gap(cfg):
    train, test = make_raw_series()
    late = [
        (date.fromisoformat(d) + timedelta(days=1)).isoformat()
        for d in test["dt"].to_pylist()
    ]
    shifted = test.set_column(test.column_names.index("dt"), "dt", pa.array(late))
    assert "gaps" in _fail(cfg, train, shifted)
