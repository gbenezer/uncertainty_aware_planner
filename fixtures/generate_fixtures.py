"""
fixtures/generate_fixtures.py

Writes one small SYNTHETIC parquet file per shared table (CONTRACTS.md
section 2), validated against contracts/schemas.py. Everything downstream can
be built and tested against these before any real model has produced output.

    python fixtures/generate_fixtures.py                 # write to fixtures/
    python fixtures/generate_fixtures.py --out /tmp/fx   # write elsewhere
    python fixtures/generate_fixtures.py --seed 7        # different synthetic world
    python fixtures/generate_fixtures.py --no-validate   # skip schema validation

Output is deterministic for a given seed, so CI can regenerate and
`git diff --exit-code fixtures/` to catch schema drift.

IMPORTANT
---------
- These values are random noise shaped to satisfy the contract. They say
  nothing about real demand, and no real dataset rows are used (CONTRACTS.md
  section 8). Dates are deliberately synthetic (2020).
- Metric values in `metrics` are made up; do not read anything into them.

Synthetic world
---------------
3 stores x 4 products = 12 series, 28 days, 24 hours/day:

    days  0-13  split "train"
    days 14-20  split "calib"   (the held-out calibration week)
    days 21-27  split "test"

Two forecast origins per series: day 13 (targets fall in the calib week,
split "calib") and day 20 (targets fall in the test week, split "test").
Two series are low-selling (mean scaled demand < 1). Roughly a quarter of
series-days contain a short stockout. In the calib week, ~5% of in-stock
hours are masked and imputed to give `imputed_hourly` rows with non-null
`actual` (used for imputation calibration and evaluation).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent

# ---- synthetic world ------------------------------------------------------

N_STORES = 3
N_PRODUCTS = 4
N_DAYS = 28
TRAIN_DAYS, CALIB_DAYS = 14, 7  # remaining 7 days are "test"
START = pd.Timestamp("2020-01-06")  # clearly synthetic
LOW_SELLING_SERIES = {2: 0.45, 9: 0.80}  # series index -> mean scaled demand (< 1)
DAILY_SCALE = 10.0  # daily demand mean = mean_scaled_demand * DAILY_SCALE
MASK_RATE = 0.05  # share of in-stock calib-week hours masked for imputation eval

IMP_MODEL = "imp-dlinear-quantile"
IMP_BASELINE = "imp-dlinear-baseline"
FC_QUANTILE = "fc-dlinear-quantile"
FC_BASELINE = "fc-dlinear-baseline"
IMP_RUN_ID = "imp-dlinear-quantile_s0_f0"
RUN_ID = "fixture-run-0001"
SEED, FOLD = 0, 0

COVARIATES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precipitation",
    "temp_avg",
    "humidity",
    "wind_level",
]
KEYS = ["store_id", "product_id", "date"]

_h = np.arange(24)
PROFILE = np.exp(-0.5 * ((_h - 12) / 5.0) ** 2) + 0.05
PROFILE = PROFILE / PROFILE.sum()  # share of daily demand per hour


def _day_split(day_idx: np.ndarray) -> np.ndarray:
    return np.where(
        day_idx < TRAIN_DAYS,
        "train",
        np.where(day_idx < TRAIN_DAYS + CALIB_DAYS, "calib", "test"),
    )


def _day(i: int) -> pd.Timestamp:
    return START + pd.Timedelta(days=int(i))


def _dt(x):
    """Datetime64[ns] for a scalar, array, or Series (scalars stay Timestamps)."""
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x)
    return pd.to_datetime(x).astype("datetime64[ns]")


# ---- builders -------------------------------------------------------------


def _series_meta(rng: np.random.Generator) -> pd.DataFrame:
    stores = [f"s{i:03d}" for i in range(1, N_STORES + 1)]
    products = [f"p{j:03d}" for j in range(1, N_PRODUCTS + 1)]
    meta = pd.DataFrame(
        [(s, p) for s in stores for p in products], columns=["store_id", "product_id"]
    )
    lam = rng.uniform(1.5, 6.0, len(meta))
    for idx, v in LOW_SELLING_SERIES.items():
        lam[idx] = v
    meta["mean_scaled_demand"] = lam
    meta["city_id"] = ["c%02d" % (1 + (int(s[1:]) % 2)) for s in meta["store_id"]]
    meta["group_id"] = ["g%02d" % (1 + (int(p[1:]) % 2)) for p in meta["product_id"]]
    return meta


def _build_hourly(meta: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_s = len(meta)
    si = np.repeat(np.arange(n_s), N_DAYS * 24)
    di = np.tile(np.repeat(np.arange(N_DAYS), 24), n_s)
    hi = np.tile(np.arange(24), n_s * N_DAYS)

    lam = meta["mean_scaled_demand"].to_numpy()
    rate = lam[si] * DAILY_SCALE * PROFILE[hi]
    sales = rng.poisson(rate).astype(float)

    # short stockout runs on ~25% of series-days
    has_so = rng.random((n_s, N_DAYS)) < 0.25
    h0 = rng.integers(8, 18, (n_s, N_DAYS))
    length = rng.integers(2, 6, (n_s, N_DAYS))
    is_so = has_so[si, di] & (hi >= h0[si, di]) & (hi < h0[si, di] + length[si, di])
    sales = np.where(is_so, 0.0, sales)  # censored: recorded as zero

    # daily covariates
    store_codes = pd.factorize(meta["store_id"])[0]
    discount = rng.choice([1.0, 0.9, 0.8], size=(n_s, N_DAYS), p=[0.6, 0.25, 0.15])
    activity = (rng.random((n_s, N_DAYS)) < 0.2).astype(int)
    holiday = np.zeros(N_DAYS, dtype=int)
    holiday[[5, 19]] = 1
    precip = rng.gamma(1.0, 2.0, (N_STORES, N_DAYS)) * (
        rng.random((N_STORES, N_DAYS)) < 0.3
    )
    temp = rng.uniform(5, 25, (N_STORES, N_DAYS))
    humid = rng.uniform(40, 90, (N_STORES, N_DAYS))
    wind = rng.integers(1, 6, (N_STORES, N_DAYS)).astype(float)
    sc = store_codes[si]

    df = pd.DataFrame(
        {
            "store_id": meta["store_id"].to_numpy()[si],
            "product_id": meta["product_id"].to_numpy()[si],
            "city_id": meta["city_id"].to_numpy()[si],
            "group_id": meta["group_id"].to_numpy()[si],
            "date": _dt(START + pd.to_timedelta(di, unit="D")),
            "hour": hi.astype("int8"),
            "sales": sales,
            "is_stockout": is_so,
            "discount": discount[si, di],
            "holiday_flag": pd.array(holiday[di], dtype="Int8"),
            "activity_flag": pd.array(activity[si, di], dtype="Int8"),
            "precipitation": precip[sc, di],
            "temp_avg": temp[sc, di],
            "humidity": humid[sc, di],
            "wind_level": wind[sc, di],
            "split": _day_split(di),
            "fold": np.int8(FOLD),
            "_rate": rate,  # internal: expected hourly demand; dropped before writing
        }
    )
    return df


def _build_imputed(hourly: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    real = hourly[hourly["is_stockout"]].copy()
    real["actual"] = np.nan  # real stockout: no ground truth

    pool = hourly[(~hourly["is_stockout"]) & (hourly["split"] == "calib")]
    masked = pool[rng.random(len(pool)) < MASK_RATE].copy()
    masked["actual"] = masked["sales"]  # masked in-stock hour: truth is known

    rows = pd.concat([real, masked], ignore_index=True)
    rate = rows["_rate"].to_numpy()
    point = rate * rng.uniform(0.85, 1.15, len(rows))
    w = 1.96 * np.sqrt(rate + 0.5)
    q025 = np.clip(point - w, 0.0, None)
    q975 = point + w
    out = pd.DataFrame(
        {
            "store_id": rows["store_id"],
            "product_id": rows["product_id"],
            "date": rows["date"],
            "hour": rows["hour"],
            "model_id": IMP_MODEL,
            "seed": SEED,
            "split": rows["split"],
            "fold": np.int8(FOLD),
            "q025": q025,
            "q975": q975,
            "point": point,
            "actual": rows["actual"],
            "q025_cal": np.clip(q025 - 0.2 * w, 0.0, None),  # post-conformal state
            "q975_cal": q975 + 0.2 * w,
        }
    )
    # Non-quantile baseline over the same hours: point estimate only, no interval (O2)
    base = out.copy()
    base["model_id"] = IMP_BASELINE
    base["point"] = rate * rng.uniform(0.8, 1.2, len(rows))
    for c in ["q025", "q975", "q025_cal", "q975_cal"]:
        base[c] = np.nan
    out = pd.concat([out, base], ignore_index=True)
    return out.sort_values(
        ["store_id", "product_id", "date", "hour", "model_id"]
    ).reset_index(drop=True)


def _build_demand_daily(hourly: pd.DataFrame, imputed: pd.DataFrame) -> pd.DataFrame:
    # one imputation run feeds demand_daily; masked eval rows (non-null actual) never do
    real = imputed[imputed["actual"].isna() & (imputed["model_id"] == IMP_MODEL)]
    obs = (
        hourly[~hourly["is_stockout"]]
        .groupby(KEYS)["sales"]
        .sum()
        .rename("demand_observed")
    )
    nso = hourly.groupby(KEYS)["is_stockout"].sum().rename("n_stockout_hours")
    rec = real.groupby(KEYS).agg(
        demand_recovered=("point", "sum"),
        recovered_lo=("q025", "sum"),
        recovered_hi=("q975", "sum"),
    )
    cov = hourly.groupby(KEYS)[COVARIATES + ["split"]].first()
    d = pd.concat([obs, nso, rec, cov], axis=1).reset_index()
    for c in ["demand_observed", "demand_recovered", "recovered_lo", "recovered_hi"]:
        d[c] = d[c].fillna(0.0)
    d["demand_total"] = d["demand_observed"] + d["demand_recovered"]
    d["n_stockout_hours"] = d["n_stockout_hours"].astype("int8")
    d["imputation_run_id"] = IMP_RUN_ID
    d["fold"] = np.int8(FOLD)
    cols = (
        KEYS
        + [
            "imputation_run_id",
            "demand_observed",
            "demand_recovered",
            "demand_total",
            "n_stockout_hours",
            "recovered_lo",
            "recovered_hi",
        ]
        + COVARIATES
        + ["split", "fold"]
    )
    return d[cols]


def _build_forecast(
    meta: pd.DataFrame,
    daily: pd.DataFrame,
    model_id: str,
    quantile: bool,
    rng: np.random.Generator,
) -> pd.DataFrame:
    actual = daily.set_index(KEYS)["demand_total"]
    frames = []
    for origin_day, split in [
        (TRAIN_DAYS - 1, "calib"),
        (TRAIN_DAYS + CALIB_DAYS - 1, "test"),
    ]:
        base = meta[["store_id", "product_id", "mean_scaled_demand"]].merge(
            pd.DataFrame({"horizon": np.arange(1, 8, dtype="uint8")}), how="cross"
        )
        base["origin_date"] = _dt(_day(origin_day))
        base["target_date"] = _dt(
            base["origin_date"] + pd.to_timedelta(base["horizon"].astype(int), unit="D")
        )
        base["split"] = split
        frames.append(base)
    f = pd.concat(frames, ignore_index=True)

    mean = f["mean_scaled_demand"].to_numpy() * DAILY_SCALE
    spread = 0.15 if quantile else 0.20
    point = mean * rng.uniform(1 - spread, 1 + spread, len(f))
    if quantile:
        w = 1.96 * np.sqrt(mean) * 0.8
        f["q025"] = np.clip(point - w, 0.0, None)
        f["q975"] = point + w
    else:  # baseline is not a quantile model: no interval (open decision O2)
        f["q025"] = np.nan
        f["q975"] = np.nan
    f["point"] = point
    idx = pd.MultiIndex.from_frame(
        f[["store_id", "product_id", "target_date"]].rename(
            columns={"target_date": "date"}
        )
    )
    f["actual"] = actual.reindex(idx).to_numpy()
    f["model_id"] = model_id
    f["seed"] = SEED
    f["fold"] = np.int8(FOLD)
    f["imputation_run_id"] = IMP_RUN_ID
    cols = [
        "store_id",
        "product_id",
        "origin_date",
        "horizon",
        "target_date",
        "model_id",
        "seed",
        "split",
        "fold",
        "imputation_run_id",
        "q025",
        "q975",
        "point",
        "actual",
    ]
    return (
        f[cols]
        .sort_values(["store_id", "product_id", "origin_date", "horizon"])
        .reset_index(drop=True)
    )


def _build_forecast_calibrated(
    forecast_q: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    f = forecast_q[
        forecast_q["split"] == "test"
    ].copy()  # calib-week forecasts are the calibration data
    qhat = rng.uniform(0.0, 1.5, len(f))
    f["lo95"] = np.clip(f["q025"] - qhat, 0.0, None)
    f["hi95"] = f["q975"] + qhat
    f["calib_window_start"] = _dt(_day(TRAIN_DAYS))
    f["calib_window_end"] = _dt(
        _day(TRAIN_DAYS + CALIB_DAYS - 1)
    )  # may equal origin_date
    f["qhat95"] = qhat
    return f.reset_index(drop=True)


def _build_review_output(
    daily: pd.DataFrame, fcal: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    cov_cols = COVARIATES
    # history: the last 14 days before the forecast origin
    h = daily[
        (daily["date"] >= _day(7))
        & (daily["date"] <= _day(TRAIN_DAYS + CALIB_DAYS - 1))
    ].copy()
    recovered = h["n_stockout_hours"] > 0
    hist = pd.DataFrame(
        {
            "store_id": h["store_id"],
            "product_id": h["product_id"],
            "date": h["date"],
            "kind": np.where(recovered, "recovered", "observed"),
            "point": h["demand_total"],
            # recovered days always carry a range (display rule); plain observed days do not
            "lo95": np.where(
                recovered, h["demand_observed"] + h["recovered_lo"], np.nan
            ),
            "hi95": np.where(
                recovered, h["demand_observed"] + h["recovered_hi"], np.nan
            ),
            "n_stockout_hours": h["n_stockout_hours"],
            "change_warning": False,
            "review_flag": False,
        }
    )
    hist[cov_cols] = h[cov_cols]  # same index as h

    fut = pd.DataFrame(
        {
            "store_id": fcal["store_id"],
            "product_id": fcal["product_id"],
            "date": fcal["target_date"],
            "kind": "forecast",
            "point": fcal["point"],
            "lo95": fcal["lo95"],
            "hi95": fcal["hi95"],
            "n_stockout_hours": np.int8(0),
        }
    )
    # planned future covariates come from the same daily table
    cov = daily.set_index(KEYS)[cov_cols]
    fut = fut.reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(fut[KEYS])
    fut[cov_cols] = cov.reindex(idx).reset_index(drop=True)

    width = fut["hi95"] - fut["lo95"]
    wide = width > width.quantile(0.75)
    changed_series = meta.iloc[3][
        ["store_id", "product_id"]
    ]  # one series gets a change warning
    changed = (fut["store_id"] == changed_series["store_id"]) & (
        fut["product_id"] == changed_series["product_id"]
    )
    fut["change_warning"] = changed.to_numpy()
    fut["review_flag"] = (wide | changed).to_numpy()

    out = pd.concat([hist, fut], ignore_index=True)
    out["model_id"] = FC_QUANTILE
    out["run_id"] = RUN_ID
    out["n_stockout_hours"] = out["n_stockout_hours"].astype("int8")
    cols = (
        KEYS
        + [
            "kind",
            "point",
            "lo95",
            "hi95",
            "n_stockout_hours",
            "change_warning",
            "review_flag",
            "model_id",
            "run_id",
        ]
        + cov_cols
    )
    return out[cols].sort_values(KEYS).reset_index(drop=True)


def _build_edit_log(daily: pd.DataFrame) -> pd.DataFrame:
    def demand(store, product, day):
        return float(
            daily.set_index(KEYS).loc[(store, product, _day(day)), "demand_total"]
        )

    def covariate(store, product, day, col):
        return float(daily.set_index(KEYS).loc[(store, product, _day(day)), col])

    edits = [
        (
            "s001",
            "p001",
            10,
            "demand",
            demand("s001", "p001", 10),
            None,
            "Known supplier outage; recorded demand understated",
            RUN_ID,
        ),
        (
            "s001",
            "p002",
            22,
            "discount",
            covariate("s001", "p002", 22, "discount"),
            "promo",
            "Promotion planned for this day",
            None,
        ),
        (
            "s002",
            "p003",
            12,
            "demand",
            demand("s002", "p003", 12),
            None,
            "Data entry error corrected by planner",
            RUN_ID,
        ),
    ]
    rows = []
    for i, (s, p, day, field, old, new, reason, applied) in enumerate(edits):
        if new is None:
            new = round(old * 1.2 + 1.0, 4)
        elif new == "promo":  # deeper discount than whatever the day already had
            new = 0.8 if abs(old - 0.8) > 1e-9 else 0.7
        rows.append(
            {
                "edit_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"fixture-edit-{i}")),
                "store_id": s,
                "product_id": p,
                "date": _dt(_day(day)),
                "field": field,
                "old_value": old,
                "new_value": float(new),
                "edited_at": pd.Timestamp("2020-02-10 14:00", tz="UTC")
                + pd.Timedelta(hours=i),
                "reason": reason,
                "applied_in_run_id": applied,
            }
        )
    df = pd.DataFrame(rows)
    df["edited_at"] = pd.to_datetime(df["edited_at"], utc=True)
    return df


def _build_slices(meta: pd.DataFrame) -> pd.DataFrame:
    s = meta[["store_id", "product_id", "mean_scaled_demand"]].copy()
    s["slice"] = np.where(s["mean_scaled_demand"] < 1.0, "low_sales", "high_sales")
    return s


def _build_metrics(
    slices: pd.DataFrame, imputed: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    low = set(
        map(
            tuple,
            slices.loc[
                slices["slice"] == "low_sales", ["store_id", "product_id"]
            ].to_numpy(),
        )
    )

    def n_series(sl):
        return {
            "all": len(slices),
            "low_sales": len(low),
            "high_sales": len(slices) - len(low),
        }[sl]

    masked = imputed[(imputed["model_id"] == IMP_MODEL) & imputed["actual"].notna()]
    is_low = [(s, p) in low for s, p in zip(masked["store_id"], masked["product_id"])]
    masked_n = {
        "all": len(masked),
        "low_sales": int(np.sum(is_low)),
        "high_sales": int(len(masked) - np.sum(is_low)),
    }

    # (feature, model_id, split, calibrated?, quantile model?)
    specs = [
        ("imputation", IMP_BASELINE, "calib", False, False),
        ("imputation", IMP_MODEL, "calib", False, True),
        ("imputation", IMP_MODEL, "calib", True, True),
        ("forecasting", FC_BASELINE, "test", False, False),
        ("forecasting", FC_QUANTILE, "test", False, True),
        ("forecasting", FC_QUANTILE, "test", True, True),
    ]
    rows = []
    for feature, model_id, split, calibrated, quantile in specs:
        for sl in ["all", "low_sales", "high_sales"]:
            n = masked_n[sl] if feature == "imputation" else n_series(sl) * 7
            penalty = 8.0 if sl == "low_sales" else 0.0  # low sellers have higher error
            vals = {
                "wape": rng.uniform(26, 34) + penalty,
                "wpe": rng.uniform(-3, 5),
                "mase": rng.uniform(0.8, 1.3),
            }
            if quantile:
                vals["picp95"] = (
                    rng.uniform(0.93, 0.97) if calibrated else rng.uniform(0.80, 0.90)
                )
                vals["mpiw95"] = (
                    rng.uniform(10, 14) if calibrated else rng.uniform(8, 12)
                )
            for metric, v in vals.items():
                rows.append(
                    {
                        "feature": feature,
                        "model_id": model_id,
                        "fold": np.int8(FOLD),
                        "seed": SEED,
                        "split": split,
                        "slice": sl,
                        "metric": metric,
                        "value": float(v) if n > 0 else np.nan,
                        "n": int(n),
                        "calibrated": calibrated,
                    }
                )
    return pd.DataFrame(rows)


# ---- public API -----------------------------------------------------------


def build_all(seed: int = 0) -> dict[str, pd.DataFrame]:
    """Return every fixture table, keyed by table name (unvalidated)."""
    rng = np.random.default_rng(seed)
    meta = _series_meta(rng)
    hourly_full = _build_hourly(meta, rng)
    imputed = _build_imputed(hourly_full, rng)
    hourly = hourly_full.drop(columns="_rate")
    daily = _build_demand_daily(hourly_full, imputed)
    fc_base = _build_forecast(meta, daily, FC_BASELINE, quantile=False, rng=rng)
    fc_quant = _build_forecast(meta, daily, FC_QUANTILE, quantile=True, rng=rng)
    forecast = pd.concat([fc_base, fc_quant], ignore_index=True)
    fcal = _build_forecast_calibrated(fc_quant, rng)
    slices = _build_slices(meta)
    return {
        "series_hourly": hourly,
        "imputed_hourly": imputed,
        "demand_daily": daily,
        "forecast": forecast,
        "forecast_calibrated": fcal,
        "review_output": _build_review_output(daily, fcal, meta),
        "edit_log": _build_edit_log(daily),
        "metrics": _build_metrics(slices, imputed, rng),
        "slices": slices,
    }


def _load_validator():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from contracts.schemas import validate  # noqa: WPS433
    except ImportError as e:
        print(
            f"WARNING: cannot import contracts.schemas ({e}); writing UNVALIDATED fixtures.",
            file=sys.stderr,
        )
        return None
    return validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=FIXTURE_DIR, help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-validate", action="store_true", help="skip schema validation")
    args = ap.parse_args(argv)

    tables = build_all(args.seed)
    validate = None if args.no_validate else _load_validator()
    args.out.mkdir(parents=True, exist_ok=True)

    failed = False
    for name, df in tables.items():
        if validate is not None:
            try:
                df = validate(name, df)
            except Exception as e:  # noqa: BLE001
                failed = True
                print(f"INVALID {name}:\n{e}\n", file=sys.stderr)
                continue
        df.to_parquet(args.out / f"{name}.parquet", index=False)
        print(f"wrote {name:20s} {len(df):6d} rows")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
