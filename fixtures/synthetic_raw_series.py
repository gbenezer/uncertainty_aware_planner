"""SYNTHETIC raw_series tables (CONTRACTS.md 2.1) for tests and offline development. Not real data.

Same schema and calendar as FreshRetailNet-50K (train 2024-03-28..06-25, eval 06-26..07-02), with
planted end-of-day stockouts, flagged night hours, a partial first stockout hour, and both discount
edge cases (0.0 missing-value code and a value above 1).

    uv run python fixtures/synthetic_raw_series.py   # rewrites fixtures/raw_series/{train,eval}.parquet
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FIXTURE_DIR = Path(__file__).resolve().parent / "raw_series"
FIRST_DAY = date(2024, 3, 28)
TRAIN_DAYS, TEST_DAYS = 90, 7
N_STORES, N_PRODUCTS = 2, 3
HOURS = 24

SCHEMA = pa.schema(
    [
        ("city_id", pa.int64()),
        ("store_id", pa.int64()),
        ("management_group_id", pa.int64()),
        ("first_category_id", pa.int64()),
        ("second_category_id", pa.int64()),
        ("third_category_id", pa.int64()),
        ("product_id", pa.int64()),
        ("dt", pa.string()),
        ("sale_amount", pa.float64()),
        ("hours_sale", pa.list_(pa.float64())),
        ("stock_hour6_22_cnt", pa.int32()),
        ("hours_stock_status", pa.list_(pa.int64())),
        ("discount", pa.float64()),
        ("holiday_flag", pa.int32()),
        ("activity_flag", pa.int32()),
        ("precpt", pa.float64()),
        ("avg_temperature", pa.float64()),
        ("avg_humidity", pa.float64()),
        ("avg_wind_level", pa.float64()),
    ]
)

# Day positions of the planted discount edge cases in series (store 0, product 0).
DISCOUNT_ZERO_DAYS = [0, 1, 2, 91]  # three train rows and one eval row
DISCOUNT_ABOVE_ONE_DAY = 3


def make_raw_series(seed: int = 0) -> tuple[pa.Table, pa.Table]:
    """Return (train, eval) tables for N_STORES x N_PRODUCTS series."""
    rng = np.random.default_rng(seed)
    n_days = TRAIN_DAYS + TEST_DAYS
    days = [FIRST_DAY + timedelta(days=i) for i in range(n_days)]
    weekend = np.array([d.weekday() >= 5 for d in days])
    profile = np.exp(-0.5 * ((np.arange(HOURS) - 9) / 2) ** 2) + np.exp(
        -0.5 * ((np.arange(HOURS) - 16) / 2.5) ** 2
    )
    profile[:6] *= 0.05
    profile[22:] *= 0.1

    rows = []
    for store in range(N_STORES):
        for product in range(N_PRODUCTS):
            level = rng.uniform(0.2, 1.5)
            discount = np.where(rng.random(n_days) < 0.3, 0.8, 1.0)
            if (store, product) == (0, 0):
                discount[DISCOUNT_ZERO_DAYS] = 0.0
                discount[DISCOUNT_ABOVE_ONE_DAY] = 1.05
            uplift = (1 + 0.27 * weekend) * np.where(discount < 1, 1.4, 1.0)
            sales = rng.poisson(10 * level * uplift[:, None] * profile[None, :]) / 10
            stock = np.zeros((n_days, HOURS), dtype=np.int64)
            sell_out = rng.random(n_days) < 0.35
            start = rng.integers(14, 21, size=n_days)
            for d in np.flatnonzero(sell_out):
                stock[d, start[d] :] = 1
                sales[d, start[d] + 1 :] = (
                    0.0  # the first flagged hour may keep partial sales
                )
            night = rng.random(n_days) < 0.1
            stock[night, :6] = 1
            sales[night, :6] = 0.0
            temperature = (
                15 + 15 * np.arange(n_days) / n_days + rng.normal(0, 1, n_days)
            )
            for d, day in enumerate(days):
                rows.append(
                    {
                        "city_id": store,
                        "store_id": store,
                        "management_group_id": product % 2,
                        "first_category_id": 10 + product % 2,
                        "second_category_id": 20 + product,
                        "third_category_id": 30 + product,
                        "product_id": 100 + product,
                        "dt": day.isoformat(),
                        "sale_amount": float(sales[d].sum()),
                        "hours_sale": sales[d].tolist(),
                        "stock_hour6_22_cnt": int(stock[d, 6:22].sum()),
                        "hours_stock_status": stock[d].tolist(),
                        "discount": float(discount[d]),
                        "holiday_flag": int(weekend[d]),
                        "activity_flag": int(0 < discount[d] < 1),
                        "precpt": float(max(0.0, rng.normal(2, 3))),
                        "avg_temperature": float(temperature[d]),
                        "avg_humidity": float(rng.uniform(50, 95)),
                        "avg_wind_level": float(rng.uniform(1, 3.5)),
                    }
                )
    rng.shuffle(rows)  # the real files are not sorted
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    is_test = pc.greater_equal(
        table["dt"], (FIRST_DAY + timedelta(days=TRAIN_DAYS)).isoformat()
    )
    return table.filter(pc.invert(is_test)), table.filter(is_test)


def fixture_config(work_dir: str | Path) -> dict:
    """The real configs/data.yaml (and splits.yaml), pointed at the committed fixture files.

    Loading still goes through HF `datasets`; all outputs and caches go under work_dir.
    """
    from uncertainty_aware_planner.data.config import load_config

    work_dir = Path(work_dir)
    cfg = load_config()
    cfg["source"] = {
        "kind": "parquet",
        "data_files": {
            "train": str(FIXTURE_DIR / "train.parquet"),
            "eval": str(FIXTURE_DIR / "eval.parquet"),
        },
    }
    cfg["paths"] = {
        "cache_dir": str(work_dir / "hf_cache"),
        "raw_dir": str(work_dir / "raw"),
        "processed_dir": str(work_dir / "processed"),
    }
    cfg["expected"] = {
        "n_series": N_STORES * N_PRODUCTS,
        "train_days": TRAIN_DAYS,
        "test_days": TEST_DAYS,
    }
    return cfg


def main() -> None:
    FIXTURE_DIR.mkdir(exist_ok=True)
    train, test = make_raw_series()
    pq.write_table(train, FIXTURE_DIR / "train.parquet")
    pq.write_table(test, FIXTURE_DIR / "eval.parquet")
    print(
        f"wrote {train.num_rows} train and {test.num_rows} eval rows to {FIXTURE_DIR}"
    )


if __name__ == "__main__":
    main()
