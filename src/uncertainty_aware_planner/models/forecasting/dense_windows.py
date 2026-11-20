"""Daily series as dense [series, day] arrays, and the forecasting windows the DLinear models use.

A light alternative to the pytorch-forecasting datasets in data/timeseries_dataset.py (about 0.1 GB instead of
about 13 GB for the full data) with the same windows, which tests/models/forecasting checks:

    training windows lie entirely inside the train window (stride 1; a fold has one window per series)
    evaluation windows start inside the eval window and read their lookback from the days just before it
    (fold val: rolling origins across the 4 weeks; final calib/test: one origin per series)

Day 0 is the first day of the final train window, so a window's first forecast day is also its time_idx.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.splits import Window

KEYS = ["store_id", "product_id"]


@dataclass(frozen=True)
class DenseDays:
    keys: pd.DataFrame  # store_id, product_id (int64): one row per series, sorted
    start: date  # day 0
    target: (
        np.ndarray
    )  # float64 [series, day]; NaN where there is no value (or it is hidden)
    in_stock: np.ndarray  # bool [series, day]: no stockout hour between 6:00 and 22:00
    covariates: dict[str, np.ndarray]  # name -> float64 [series, day]

    @property
    def n_series(self) -> int:
        return self.target.shape[0]

    @property
    def n_days(self) -> int:
        return self.target.shape[1]

    def day(self, when: date) -> int:
        return (when - self.start).days

    def hide_targets_from(self, day: int) -> DenseDays:
        """Copy with every target on and after `day` set to NaN, so reading one raises (no leakage)."""
        target = self.target.copy()
        target[:, day:] = np.nan
        return replace(self, target=target)


def dense_days(
    demand_daily: pd.DataFrame,
    start: date,
    target: str = "demand_total",
    covariates: Sequence[str] = (),
) -> DenseDays:
    """Arrays from demand_daily (CONTRACTS.md 2.4), which must hold one row per series and day from `start`."""
    columns = KEYS + ["date", target, "stock_hour6_22_cnt", *covariates]
    df = demand_daily[list(dict.fromkeys(columns))]
    day = (pd.to_datetime(df["date"]) - pd.Timestamp(start)).dt.days.to_numpy()
    store = df["store_id"].to_numpy(dtype="int64")
    product = df["product_id"].to_numpy(dtype="int64")
    order = np.lexsort((day, product, store))
    day, store, product = day[order], store[order], product[order]

    n_days = int(day.max()) + 1
    n_series, rest = divmod(len(df), n_days)
    if day.min() != 0 or rest:
        raise ValueError(
            "demand_daily must hold one row per series and day from the start"
        )
    day, store, product = (a.reshape(n_series, n_days) for a in (day, store, product))
    if not (
        (day == np.arange(n_days)).all()
        and (store == store[:, :1]).all()
        and (product == product[:, :1]).all()
    ):
        raise ValueError(
            "demand_daily must hold one row per series and day from the start"
        )

    def grid(column: str) -> np.ndarray:
        values = df[column].to_numpy(dtype="float64", na_value=np.nan)
        return values[order].reshape(n_series, n_days)

    return DenseDays(
        keys=pd.DataFrame({"store_id": store[:, 0], "product_id": product[:, 0]}),
        start=start,
        target=grid(target),
        in_stock=grid("stock_hour6_22_cnt") == 0,
        covariates={name: grid(name) for name in covariates},
    )


@dataclass(frozen=True)
class Windows:
    series: np.ndarray  # int64 [n]: row in DenseDays
    first_day: np.ndarray  # int64 [n]: first forecast day (= time_idx)
    lookback: int
    horizon: int

    def __len__(self) -> int:
        return len(self.series)


def _windows(n_series: int, first: int, last: int, lookback: int, horizon: int):
    firsts = np.arange(first, last + 1, dtype="int64")
    return Windows(
        series=np.repeat(np.arange(n_series, dtype="int64"), len(firsts)),
        first_day=np.tile(firsts, n_series),
        lookback=lookback,
        horizon=horizon,
    )


def train_windows(
    days: DenseDays, window: Window, lookback: int, horizon: int
) -> Windows:
    """Every stride-1 window whose lookback and horizon both lie inside `window`."""
    first = days.day(window.start) + lookback
    last = days.day(window.end) - horizon + 1
    if last < first:
        raise ValueError(
            f"{window.days} train days cannot hold a {lookback}-day lookback and a {horizon}-day horizon"
        )
    return _windows(days.n_series, first, last, lookback, horizon)


def eval_windows(
    days: DenseDays, window: Window, lookback: int, horizon: int
) -> Windows:
    """Every window whose horizon lies inside `window`; the lookback may reach back before it."""
    first = days.day(window.start)
    last = days.day(window.end) - horizon + 1
    if first - lookback < 0 or last < first:
        raise ValueError(
            f"cannot forecast {window.start}..{window.end} with a {lookback}-day lookback"
        )
    return _windows(days.n_series, first, last, lookback, horizon)


def gather(values: np.ndarray, w: Windows, offset: int, length: int) -> np.ndarray:
    """values[series, first_day + offset + 0..length-1] for every window, as [n, length]. Raises on NaN, so a
    window that would read a missing or hidden value fails loudly."""
    columns = w.first_day[:, None] + offset + np.arange(length)[None, :]
    if columns.min() < 0 or columns.max() >= values.shape[1]:
        raise ValueError("a window reaches outside the data")
    out = values[w.series[:, None], columns]
    if out.dtype.kind == "f" and np.isnan(out).any():
        raise ValueError("a window reads missing (or hidden) values")
    return out


def index_frame(days: DenseDays, w: Windows) -> pd.DataFrame:
    """time_idx (first forecast day) and group ids per window, as forecast_table.long_quantiles expects."""
    return pd.DataFrame(
        {
            "time_idx": w.first_day,
            "store_id": days.keys["store_id"].to_numpy()[w.series],
            "product_id": days.keys["product_id"].to_numpy()[w.series],
        }
    )
