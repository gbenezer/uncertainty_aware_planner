"""Forecast metrics for CONTRACTS.md 2.5 forecast tables, in the 2.10 long format.

Every run, whatever demand history it was trained on, is scored against raw recorded sales (`demand_total` of
the censored demand_daily), and only target days with no stockout between 06:00 and 22:00 are scored, as the
dataset authors do (experiments/vishnu/paper_wape.py).

WAPE and WPE follow the literature and Dev Contract A.4: one ratio over all scored rows, each row's error taken
on its own store-product-day. The dataset paper's own scoring (the ratio per (origin_date, target_date) cell,
averaged over the cells; for the test week the mean of the 7 daily values) is kept as `paper_wape` to compare
with its published numbers. PICP and MPIW use the team functions in evaluation/uncertainty_metrics.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from uncertainty_aware_planner.data.splits import Window, final_windows, fold_windows
from uncertainty_aware_planner.evaluation.uncertainty_metrics import (
    mean_prediction_interval_width,
    prediction_interval_coverage_fraction,
)

FEATURE = "forecasting"
CENSORED_RUN = "imp-none-censored"
KEYS = ["store_id", "product_id"]
ROW_KEY = KEYS + ["origin_date", "horizon"]
TRUTH_COLUMNS = KEYS + ["date", "demand_total", "stock_hour6_22_cnt"]
SLICES = ("all", "low_sales", "high_sales")
POINTS = ("estimated_demand", "q500")
PROTOCOLS = ("project", "paper")
SPLITS = ("val", "calib", "test")
BOUNDS = {False: ("q025", "q975"), True: ("q025_cal", "q975_cal")}
# psd window for the sales slices: the training days before the test week, as for the authors' psd.
SLICE_START, SLICE_END = date(2024, 3, 28), date(2024, 6, 25)
# Tolerance for actual_demand against the truth in censored runs (float32 round trips stay inside it).
ACTUAL_TOLERANCE = 1e-6

# 2.10 primary key, extended with the extra columns that tell evaluations of the same run apart.
METRIC_KEY = [
    "feature",
    "model_id",
    "imputation_run_id",
    "protocol",
    "point",
    "fold",
    "seed",
    "split",
    "slice",
    "metric",
    "calibrated",
]
# n: rows scored (wape, wpe, picp95, mpiw95) or series used (mase). n_skipped: cells with no actual demand
# (wape, wpe), series with a zero or undefined scale (mase), scored rows without both bounds (picp95, mpiw95).
METRIC_COLUMNS = [
    "feature",
    "model_id",
    "fold",
    "seed",
    "split",
    "slice",
    "metric",
    "calibrated",
    "value",
    "n",
    "imputation_run_id",
    "protocol",
    "point",
    "n_skipped",
]


def as_day(values: pd.Series) -> pd.Series:
    """Calendar dates (date objects or datetime64) as datetime64[s]."""
    return pd.to_datetime(values).astype("datetime64[s]")


def read_parquet(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Parquet columns with calendar dates read as datetime64[s] instead of date objects (less memory)."""
    missing = [column for column in columns if column not in pq.read_schema(path).names]
    if missing:
        raise ValueError(f"{path} has no {missing} columns")
    table = pq.read_table(path, columns=columns)
    for i, field in enumerate(table.schema):
        if pa.types.is_date(field.type):
            day = pc.cast(table.column(i), pa.timestamp("s"))
            table = table.set_column(i, field.name, day)
    return table.to_pandas(split_blocks=True, self_destruct=True)


def load_truth(path: str | Path) -> pd.DataFrame:
    """The scoring columns of a demand_daily file (the censored one is the truth), one row per series-day."""
    return prepare_truth(read_parquet(path, TRUTH_COLUMNS))


def prepare_truth(truth: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(TRUTH_COLUMNS) - set(truth.columns))
    if missing:
        raise ValueError(f"the truth table has no {missing} columns")
    truth = truth[TRUTH_COLUMNS].copy()
    truth["date"] = as_day(truth["date"])
    if truth.duplicated(KEYS + ["date"]).any():
        raise ValueError(
            "the truth table has duplicate (store_id, product_id, date) rows"
        )
    return truth


def build_slices(
    truth: pd.DataFrame, start: date = SLICE_START, end: date = SLICE_END
) -> pd.DataFrame:
    """One row per series: psd, the mean truth demand_total from start to end, and its slice (low_sales if
    psd < 1, else high_sales), as in the dataset authors' repository (CONTRACTS.md 2.10)."""
    days = as_day(truth["date"])
    inside = (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
    psd = truth[inside].groupby(KEYS)["demand_total"].mean().rename("psd")
    slices = truth[KEYS].drop_duplicates().merge(psd.reset_index(), on=KEYS, how="left")
    slices["slice"] = np.where(slices["psd"] < 1, "low_sales", "high_sales")
    return slices.sort_values(KEYS).reset_index(drop=True)


def training_window(splits_cfg: dict, fold: int | None, protocol: str) -> Window:
    """Training days of a run, for the MASE scale. The paper protocol trains the final model through the
    calibration week."""
    if fold is not None:
        return fold_windows(splits_cfg)[int(fold)]["train"]
    final = final_windows(splits_cfg)
    if protocol == "paper":
        return Window("train", None, final["train"].start, final["calib"].end)
    if protocol != "project":
        raise ValueError(f"protocol must be one of {PROTOCOLS}, not {protocol!r}")
    return final["train"]


def _check_rows(rows: pd.DataFrame, point: str) -> None:
    if rows["forecast"].isna().any():
        raise ValueError(
            f"{rows['forecast'].isna().sum()} rows have no {point} forecast"
        )
    for lower, upper in BOUNDS.values():
        if (lower in rows) != (upper in rows):
            raise ValueError(f"the table has only one of {lower} and {upper}")
        if lower in rows:
            half = rows[lower].isna() != rows[upper].isna()
            if half.any():
                raise ValueError(
                    f"{half.sum()} rows have only one of {lower} and {upper}"
                )
    duplicated = rows.duplicated(ROW_KEY)
    if duplicated.any():
        raise ValueError(
            f"{duplicated.sum()} duplicate (store_id, product_id, origin_date, horizon) rows"
        )
    lead = (rows["target_date"] - rows["origin_date"]) / pd.Timedelta(days=1)
    if (lead != rows["horizon"]).any():
        raise ValueError(
            "target_date must be origin_date + horizon days (CONTRACTS.md 2.5)"
        )


def score_table(
    table: pd.DataFrame,
    truth: pd.DataFrame,
    point: str = "estimated_demand",
    check_actuals: bool = False,
) -> pd.DataFrame:
    """Forecast rows joined to the truth on (store_id, product_id, target_date), keeping the scored rows: the
    target day has a recorded demand and no stockout between 06:00 and 22:00.

    `truth` holds one row per series-day (load_truth). `point` is the forecast column: estimated_demand, or
    q500 joined from the run directory. Raises on forecast rows without a truth row, duplicate rows, null
    forecasts, rows with one interval bound, a target_date other than origin_date + horizon, and
    (check_actuals, for censored runs) an actual_demand that differs from the truth. Returns store_id,
    product_id, origin_date, horizon, target_date, forecast, actual and the interval bounds the table has.
    """
    columns = ROW_KEY + ["target_date", point]
    extra = ["actual_demand"] if check_actuals else []
    missing = [column for column in columns + extra if column not in table.columns]
    if missing:
        raise ValueError(f"the forecast table has no {missing} columns")
    bounds = [c for pair in BOUNDS.values() for c in pair if c in table.columns]
    rows = table[columns + bounds + extra].rename(columns={point: "forecast"})
    rows["forecast"] = rows["forecast"].astype("float64")
    for column in ("origin_date", "target_date"):
        rows[column] = as_day(rows[column])
    _check_rows(rows, point)

    on = KEYS + ["target_date"]
    by_target = truth[TRUTH_COLUMNS].rename(columns={"date": "target_date"})
    if by_target["target_date"].dtype != "datetime64[s]":
        by_target["target_date"] = as_day(by_target["target_date"])
    # A left join keeps the row order, so the truth columns line up with `rows`.
    found = rows[on].merge(by_target, on=on, how="left", indicator=True)
    if len(found) != len(rows):
        raise ValueError(
            "the truth table has duplicate (store_id, product_id, date) rows"
        )
    unmatched = (found["_merge"] == "left_only").to_numpy()
    if unmatched.any():
        example = rows.loc[unmatched, on].iloc[0].to_dict()
        raise ValueError(
            f"{unmatched.sum()} forecast rows have no truth row, e.g. {example}"
        )
    actual = found["demand_total"].to_numpy(dtype="float64")

    if check_actuals:
        recorded = rows["actual_demand"].to_numpy(dtype="float64")
        both = ~np.isnan(recorded) & ~np.isnan(actual)
        close = np.isclose(
            recorded[both], actual[both], rtol=ACTUAL_TOLERANCE, atol=ACTUAL_TOLERANCE
        )
        if not close.all():
            raise ValueError(
                f"{(~close).sum()} rows have an actual_demand that differs from the censored truth"
            )

    keep = ~np.isnan(actual) & (found["stock_hour6_22_cnt"] == 0).to_numpy()
    scored = rows.loc[keep, ROW_KEY + ["target_date", "forecast"] + bounds]
    scored.insert(len(ROW_KEY) + 2, "actual", actual[keep])
    return scored.reset_index(drop=True)


def attach_slices(scored: pd.DataFrame, slices: pd.DataFrame) -> pd.DataFrame:
    """Add each row's sales slice from build_slices / slices.parquet; raises if a scored series has none."""
    unknown = set(slices["slice"].unique()) - {"low_sales", "high_sales"}
    if unknown:
        raise ValueError(f"unknown slices {sorted(unknown)}")
    out = scored.merge(
        slices[KEYS + ["slice"]], on=KEYS, how="left", validate="many_to_one"
    )
    if out["slice"].isna().any():
        missing = out.loc[out["slice"].isna(), KEYS].drop_duplicates()
        raise ValueError(
            f"{len(missing)} scored series have no slice; rebuild slices.parquet"
        )
    out["slice"] = pd.Categorical(out["slice"], categories=["low_sales", "high_sales"])
    return out


def _members(scored: pd.DataFrame):
    if "slice" not in scored:
        raise ValueError("scored rows need a slice column (attach_slices)")
    for name in SLICES:
        if name == "all":
            yield name, np.ones(len(scored), bool)
        else:
            yield name, (scored["slice"] == name).to_numpy(dtype=bool)


def wape_wpe(scored: pd.DataFrame) -> pd.DataFrame:
    """WAPE and WPE per slice as in the literature and Dev Contract A.4: sum|F - A| / sum|A| and
    sum(F - A) / sum|A| over all scored rows (WPE > 0 is over-forecast). Each row is one store-product-day, so
    no residuals are summed before the absolute value. A slice with no demand gives NaN."""
    error = (scored["forecast"] - scored["actual"]).to_numpy(dtype="float64")
    abs_actual = scored["actual"].abs().to_numpy(dtype="float64")
    rows = []
    for name, member in _members(scored):
        total = abs_actual[member].sum()
        used = total > 0
        for metric, values in (("wape", np.abs(error)), ("wpe", error)):
            rows.append(
                {
                    "slice": name,
                    "metric": metric,
                    "value": values[member].sum() / total if used else np.nan,
                    "n": int(member.sum()) if used else 0,
                    "n_skipped": 0 if used else int(member.sum()),
                }
            )
    return pd.DataFrame(rows)


def paper_wape_wpe(scored: pd.DataFrame) -> pd.DataFrame:
    """The dataset paper's WAPE and WPE per slice: the same ratios per (origin_date, target_date) cell,
    averaged over the cells; for the test week, the mean of the 7 daily values. Cells with sum|A| = 0 are
    skipped. Kept to compare with the published numbers."""
    error = scored["forecast"] - scored["actual"]
    frame = pd.DataFrame(
        {
            "origin_date": scored["origin_date"],
            "target_date": scored["target_date"],
            "abs_error": error.abs(),
            "error": error,
            "abs_actual": scored["actual"].abs(),
        }
    )
    rows = []
    for name, member in _members(scored):
        cells = (
            frame[member]
            .groupby(["origin_date", "target_date"])
            .agg(
                abs_error=("abs_error", "sum"),
                error=("error", "sum"),
                abs_actual=("abs_actual", "sum"),
                rows=("error", "size"),
            )
        )
        used = cells[cells["abs_actual"] > 0]
        for metric, numerator in (("wape", "abs_error"), ("wpe", "error")):
            rows.append(
                {
                    "slice": name,
                    "metric": metric,
                    "value": (used[numerator] / used["abs_actual"]).mean(),
                    "n": int(used["rows"].sum()),
                    "n_skipped": len(cells) - len(used),
                }
            )
    return pd.DataFrame(rows)


def paper_wape(scored: pd.DataFrame) -> pd.DataFrame:
    """Paper-style WAPE per slice (paper_wape_wpe), for comparison with the published numbers; not a 2.10
    metric."""
    rows = paper_wape_wpe(scored)
    rows = rows[rows["metric"] == "wape"]
    return rows[["slice", "value", "n"]].reset_index(drop=True)


def series_scale(truth: pd.DataFrame, start: date, end: date) -> pd.Series:
    """MASE scale per series: mean |A_t - A_(t-1)| over consecutive days from start to end (inclusive),
    indexed by (store_id, product_id); NaN for a series without two consecutive days."""
    days = truth["date"]
    if days.dtype != "datetime64[s]":
        days = as_day(days)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    inside = ((days >= first) & (days <= last)).to_numpy()
    part = truth.loc[inside, KEYS + ["demand_total"]]
    day = ((days[inside] - first) // pd.Timedelta(days=1)).to_numpy()
    grouped = part.groupby(KEYS, sort=True)
    series = grouped.ngroup().to_numpy()
    index = grouped.size().index

    wide = np.full((len(index), (last - first).days + 1), np.nan)
    wide[series, day] = part["demand_total"].to_numpy(dtype="float64")
    steps = np.abs(np.diff(wide, axis=1))
    valid = ~np.isnan(steps)
    count = valid.sum(axis=1)
    total = np.where(valid, steps, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = total / count
    return pd.Series(scale, index=index, name="scale")


def mase(scored: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """MASE per slice: per series, mean |A - F| over its scored rows divided by its training-window scale
    (series_scale), averaged over series. Series with a zero or undefined scale are dropped."""
    per_series = (
        scored.assign(abs_error=(scored["forecast"] - scored["actual"]).abs())
        .groupby(KEYS, observed=True)
        .agg(abs_error=("abs_error", "mean"), slice=("slice", "first"))
    )
    series_scale_ = scale.reindex(per_series.index).to_numpy(dtype="float64")
    usable = series_scale_ > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = per_series["abs_error"].to_numpy() / series_scale_
    rows = []
    for name, member in _members(per_series):
        used = member & usable
        rows.append(
            {
                "slice": name,
                "metric": "mase",
                "value": ratio[used].mean() if used.any() else np.nan,
                "n": int(used.sum()),
                "n_skipped": int((member & ~usable).sum()),
            }
        )
    return pd.DataFrame(rows)


def _sample_size(result: dict) -> int:
    for key in ("n_picp", "n_mpiw", "n"):
        if key in result:
            return int(result[key])
    return 0


def interval_metrics(scored: pd.DataFrame, calibrated: bool = False) -> pd.DataFrame:
    """PICP and MPIW of the 95% interval per slice with the team functions in evaluation/uncertainty_metrics.py:
    [q025, q975], or [q025_cal, q975_cal] when calibrated, on the scored rows (actual = raw sales). Bounds are
    inclusive; rows without both bounds and degenerate intervals (lower == upper, e.g. [0, 0]) are left out
    and counted in n_skipped."""
    lower, upper = BOUNDS[calibrated]
    frame = pd.DataFrame(
        {
            "actual_demand": scored["actual"].to_numpy(dtype="float64"),
            lower: scored[lower].to_numpy(dtype="float64"),
            upper: scored[upper].to_numpy(dtype="float64"),
        }
    )
    rows = []
    for name, member in _members(scored):
        part = frame[member]
        results = {
            "picp95": prediction_interval_coverage_fraction(
                part, conformal_intervals=calibrated
            ),
            "mpiw95": mean_prediction_interval_width(
                part, conformal_intervals=calibrated
            ),
        }
        for metric, result in results.items():
            n = _sample_size(result)
            rows.append(
                {
                    "slice": name,
                    "metric": metric,
                    "value": float(result["value"]),
                    "n": n,
                    "n_skipped": int(member.sum()) - n,
                }
            )
    return pd.DataFrame(rows)


def has_bounds(frame: pd.DataFrame, calibrated: bool = False) -> bool:
    """Whether a table carries the raw (or calibrated) interval: point-only tables leave both null."""
    lower, upper = BOUNDS[calibrated]
    return (
        lower in frame
        and upper in frame
        and bool((frame[lower].notna() & frame[upper].notna()).any())
    )


def metric_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """2.10 rows in METRIC_COLUMNS order with the contract dtypes (fold is a nullable int)."""
    out = rows[METRIC_COLUMNS].copy()
    out["fold"] = out["fold"].astype("Int64")
    out["seed"] = out["seed"].astype("int64")
    out["calibrated"] = out["calibrated"].astype(bool)
    out["value"] = out["value"].astype("float64")
    out["n"] = out["n"].astype("int64")
    out["n_skipped"] = out["n_skipped"].astype("int64")
    return out.reset_index(drop=True)


def metric_rows(
    scored: pd.DataFrame,
    scale: pd.Series,
    meta: dict,
    intervals: bool | None = None,
    calibrated: bool | None = None,
) -> pd.DataFrame:
    """All 2.10 rows of one scored table: wape, wpe and mase per slice, plus picp95 and mpiw95 for the raw
    interval and, once conformalized, the calibrated one. `meta` holds model_id, imputation_run_id,
    protocol, point, fold, seed and split; `intervals` / `calibrated` default to whether the bounds are set.
    """
    intervals = has_bounds(scored) if intervals is None else intervals
    calibrated = has_bounds(scored, True) if calibrated is None else calibrated
    parts = [
        wape_wpe(scored).assign(calibrated=False),
        mase(scored, scale).assign(calibrated=False),
    ]
    if intervals:
        parts.append(interval_metrics(scored).assign(calibrated=False))
    if calibrated:
        parts.append(interval_metrics(scored, calibrated=True).assign(calibrated=True))
    rows = pd.concat(parts, ignore_index=True).assign(feature=FEATURE, **meta)
    return metric_frame(rows)


def check_metric_keys(metrics: pd.DataFrame) -> None:
    """Raise if two metric rows share a key (METRIC_KEY)."""
    duplicated = metrics.duplicated(METRIC_KEY, keep=False)
    if duplicated.any():
        example = metrics.loc[duplicated, METRIC_KEY].iloc[0].to_dict()
        raise ValueError(f"{duplicated.sum()} metric rows share a key, e.g. {example}")


def read_metrics(path: str | Path) -> pd.DataFrame:
    """A metrics.csv written by the evaluation, with the contract dtypes."""
    return metric_frame(pd.read_csv(path, dtype={"fold": "Int64", "calibrated": bool}))


def seed_average(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean forecast per (store_id, product_id, origin_date, horizon) over the seeds in `scored`."""
    aggregations = {
        "target_date": ("target_date", "first"),
        "forecast": ("forecast", "mean"),
        "actual": ("actual", "first"),
        "n_seeds": ("forecast", "size"),
    }
    if "slice" in scored:
        aggregations["slice"] = ("slice", "first")
    return scored.groupby(ROW_KEY, observed=True).agg(**aggregations).reset_index()


def paired_bootstrap_delta(
    scored_a: pd.DataFrame, scored_b: pd.DataFrame, n_boot: int = 1000, seed: int = 0
) -> dict:
    """95% CI of the WAPE difference a - b (wape_wpe, one ratio over all rows) on the rows both models scored;
    negative favours a.

    Forecasts are first averaged over seeds per (series, origin, horizon). Each draw resamples series with
    replacement, the same draw for both models, and recomputes both ratios.
    """
    a, b = seed_average(scored_a), seed_average(scored_b)
    both = a.merge(b[ROW_KEY + ["forecast"]], on=ROW_KEY, suffixes=("_a", "_b"))
    if both.empty:
        raise ValueError("the two models share no scored rows")
    series = both.groupby(KEYS, sort=True).ngroup().to_numpy()
    n_series = series.max() + 1

    def per_series(values: pd.Series) -> np.ndarray:
        return np.bincount(
            series, weights=values.to_numpy(dtype="float64"), minlength=n_series
        )

    abs_actual = per_series(both["actual"].abs())
    error_a = per_series((both["forecast_a"] - both["actual"]).abs())
    error_b = per_series((both["forecast_b"] - both["actual"]).abs())
    wape_a = float(error_a.sum() / abs_actual.sum())
    wape_b = float(error_b.sum() / abs_actual.sum())

    rng = np.random.default_rng(seed)
    chunk = max(1, 5_000_000 // n_series)
    deltas = []
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        draws = rng.integers(0, n_series, size=(size, n_series))
        flat = (np.arange(size)[:, None] * n_series + draws).ravel()
        weights = np.bincount(flat, minlength=size * n_series).reshape(size, n_series)
        weights = weights.astype("float64")
        denominator = weights @ abs_actual
        with np.errstate(invalid="ignore", divide="ignore"):
            deltas.append(
                (weights @ error_a) / denominator - (weights @ error_b) / denominator
            )
    low, high = np.nanpercentile(np.concatenate(deltas), [2.5, 97.5])
    return {
        "wape_a": wape_a,
        "wape_b": wape_b,
        "delta": wape_a - wape_b,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_series": int(n_series),
        "n_rows": len(both),
        "n_boot": n_boot,
    }


def top_misforecasts(scored: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The k largest |F - A| of the seed-averaged forecasts, over all rows (group 'all') and within low_sales
    (group 'low_sales'). Pass the scored test-week rows of one model, all seeds."""
    rows = seed_average(scored)
    rows["error"] = rows["forecast"] - rows["actual"]
    rows["abs_error"] = rows["error"].abs()
    parts = []
    for group, part in (
        ("all", rows),
        ("low_sales", rows[rows["slice"] == "low_sales"]),
    ):
        top = part.nlargest(k, "abs_error")
        parts.append(top.assign(group=group, rank=np.arange(1, len(top) + 1)))
    out = pd.concat(parts, ignore_index=True)
    out["slice"] = out["slice"].astype(str)
    for column in ("origin_date", "target_date"):
        out[column] = out[column].dt.date
    columns = [
        "group",
        "rank",
        "store_id",
        "product_id",
        "slice",
        "origin_date",
        "target_date",
    ]
    columns += ["horizon", "forecast", "actual", "error", "abs_error", "n_seeds"]
    return out[columns]
