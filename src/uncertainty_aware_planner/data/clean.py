"""Step 2: validation and cleaning shared by every data output.

Checks follow the EDA data-quality scorecard; cleaning follows decision D2 (contract columns only).
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

HOURS = 24
SALES_TOLERANCE = 1e-6
SCALAR_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "dt",
    "sale_amount",
    "stock_hour6_22_cnt",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


class DataValidationError(ValueError):
    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__("raw data failed validation:\n  - " + "\n  - ".join(failures))


def validate_split(table: pa.Table, n_rows: int, name: str) -> list[str]:
    """Row-level checks on one raw split (raw_series, CONTRACTS.md 2.1). Returns failure messages."""
    failures = []
    missing = [
        c
        for c in SCALAR_COLUMNS + ["hours_sale", "hours_stock_status"]
        if c not in table.column_names
    ]
    if missing:
        return [f"[{name}] missing columns {missing}"]
    if table.num_rows != n_rows:
        failures.append(f"[{name}] {table.num_rows} rows, expected {n_rows}")
    for column in SCALAR_COLUMNS + ["hours_sale", "hours_stock_status"]:
        if table[column].null_count:
            failures.append(f"[{name}] {table[column].null_count} nulls in {column}")
    for column in ("hours_sale", "hours_stock_status"):
        lengths = pc.list_value_length(table[column])
        if pc.any(pc.not_equal(lengths, HOURS)).as_py():
            failures.append(f"[{name}] {column} lists must all have {HOURS} values")
    if failures:
        return failures

    stock = pc.list_flatten(table["hours_stock_status"])
    if not pc.all(pc.is_in(stock, pa.array([0, 1], stock.type))).as_py():
        failures.append(f"[{name}] hours_stock_status has values other than 0 and 1")
    hourly = pc.list_flatten(table["hours_sale"]).to_numpy().reshape(-1, HOURS)
    daily = table["sale_amount"].to_numpy()
    if (hourly < 0).any() or (daily < 0).any():
        failures.append(f"[{name}] negative sales")
    gap = np.abs(hourly.sum(axis=1) - daily).max(initial=0.0)
    if gap > SALES_TOLERANCE:
        failures.append(
            f"[{name}] hourly sales differ from sale_amount by up to {gap:.3g}"
        )
    return failures


def validate_calendar(
    store: np.ndarray,
    product: np.ndarray,
    dates: np.ndarray,
    n_series: int,
    n_days: int,
) -> list[str]:
    """On rows sorted by (store, product, date): no duplicates, and every series has the same contiguous days."""
    if len(dates) != n_series * n_days:
        return [
            f"{len(dates)} rows after combining splits, expected {n_series * n_days}"
        ]
    failures = []
    same_key = (
        (store[1:] == store[:-1])
        & (product[1:] == product[:-1])
        & (dates[1:] == dates[:-1])
    )
    if same_key.any():
        failures.append(
            f"{int(same_key.sum())} duplicate (store_id, product_id, dt) rows"
        )
    store, product, dates = (
        a.reshape(n_series, n_days) for a in (store, product, dates)
    )
    if (store != store[:, :1]).any() or (product != product[:, :1]).any():
        failures.append(f"series do not all have {n_days} days")
    if (dates != dates[:1]).any():
        failures.append("series do not share the same calendar")
    if (np.diff(dates[0]) != np.timedelta64(1, "D")).any():
        failures.append(
            "calendar has gaps (the eval split must start the day after train ends)"
        )
    return failures


def clean_discount(
    discount: np.ndarray, cleaning: dict
) -> tuple[np.ndarray, dict[str, int]]:
    """Recode the missing-value code to the fill value and cap values above the maximum (EDA Q2, Q3)."""
    is_missing = discount == cleaning["discount_missing_code"]
    is_above = discount > cleaning["discount_max"]
    cleaned = np.where(
        is_missing,
        cleaning["discount_fill"],
        np.minimum(discount, cleaning["discount_max"]),
    )
    return cleaned, {
        "discount_missing_code": int(is_missing.sum()),
        "discount_above_max": int(is_above.sum()),
    }
