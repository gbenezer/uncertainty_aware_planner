"""The cleaned dataset as dense arrays (series x day x hour). Every data output is built from a Cube."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .clean import (
    HOURS,
    DataValidationError,
    clean_discount,
    validate_calendar,
    validate_split,
)
from .load import check_raw_files, open_dataset, to_arrow, write_version_record

HIERARCHY = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]
COVARIATES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]
SORT_KEYS = [
    ("store_id", "ascending"),
    ("product_id", "ascending"),
    ("dt", "ascending"),
]


@dataclass(frozen=True)
class Cube:
    series: (
        pd.DataFrame
    )  # one row per store-product series: hierarchy IDs, sorted by store_id, product_id
    dates: np.ndarray  # (n_days,) datetime64[D]
    sales: np.ndarray  # (n_series, n_days, 24) float64 hourly scaled sales
    stockout: (
        np.ndarray
    )  # (n_series, n_days, 24) bool; True where hours_stock_status == 1 (out of stock)
    daily: pd.DataFrame  # n_series * n_days rows, series-major: hierarchy IDs, date, daily scalars, cleaned covariates
    cleaning_report: dict = field(default_factory=dict)

    @property
    def n_series(self) -> int:
        return len(self.series)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def day_index(self, day: date) -> int:
        index = int((np.datetime64(day, "D") - self.dates[0]) // np.timedelta64(1, "D"))
        if not 0 <= index < self.n_days:
            raise ValueError(f"{day} is outside the data range")
        return index

    def daily_array(self, columns: list[str], dtype=np.float64) -> np.ndarray:
        """Daily columns reshaped to (n_series, n_days, len(columns))."""
        return (
            self.daily[columns]
            .to_numpy(dtype)
            .reshape(self.n_series, self.n_days, len(columns))
        )

    def series_index(self, store_id, product_id) -> np.ndarray:
        """Row positions in `series` for the given IDs; raises if any pair is unknown."""
        keys = pd.MultiIndex.from_frame(self.series[["store_id", "product_id"]])
        positions = keys.get_indexer(
            pd.MultiIndex.from_arrays([np.asarray(store_id), np.asarray(product_id)])
        )
        if (positions < 0).any():
            raise ValueError(
                f"{int((positions < 0).sum())} rows refer to unknown store-product series"
            )
        return positions


def build_cube(train: pa.Table, test: pa.Table, cfg: dict) -> Cube:
    """Validate both raw splits, combine and sort them, and apply the shared cleaning."""
    expected = cfg["expected"]
    n_series = expected["n_series"]
    n_days = expected["train_days"] + expected["test_days"]
    failures = validate_split(train, n_series * expected["train_days"], "train")
    failures += validate_split(test, n_series * expected["test_days"], "eval")
    if failures:
        raise DataValidationError(failures)

    report = {}
    for name, table in (("train", train), ("eval", test)):
        _, report[name] = clean_discount(table["discount"].to_numpy(), cfg["cleaning"])

    table = pa.concat_tables([train, test.cast(train.schema)])
    table = table.take(pc.sort_indices(table, sort_keys=SORT_KEYS))
    store = table["store_id"].to_numpy()
    product = table["product_id"].to_numpy()
    dates = pc.cast(table["dt"], pa.date32()).to_numpy().astype("datetime64[D]")
    failures = validate_calendar(store, product, dates, n_series, n_days)
    if failures:
        raise DataValidationError(failures)

    shape = (n_series, n_days, HOURS)
    sales = (
        pc.list_flatten(table["hours_sale"])
        .to_numpy()
        .astype(np.float64)
        .reshape(shape)
    )
    stockout = (pc.list_flatten(table["hours_stock_status"]).to_numpy() == 1).reshape(
        shape
    )

    daily = table.drop_columns(["hours_sale", "hours_stock_status", "dt"]).to_pandas()
    daily.insert(len(HIERARCHY), "date", dates)
    daily["discount"], _ = clean_discount(daily["discount"].to_numpy(), cfg["cleaning"])
    series = daily[HIERARCHY].iloc[::n_days].reset_index(drop=True)
    return Cube(series, dates[:n_days].copy(), sales, stockout, daily, report)


def load_cube(cfg: dict) -> Cube:
    """Load with HF `datasets` (checking and recording the version for Hub downloads), validate, and clean."""
    dataset_dict = open_dataset(cfg)
    if cfg["source"]["kind"] == "hub":
        write_version_record(cfg, dataset_dict, check_raw_files(cfg))
    return build_cube(
        to_arrow(dataset_dict["train"]), to_arrow(dataset_dict["eval"]), cfg
    )
