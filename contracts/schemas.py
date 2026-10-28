"""
contracts/schemas.py

Enforced version of CONTRACTS.md. If this file and CONTRACTS.md disagree,
fix both in the same PR.

STATUS: DRAFT 0.1, freeze at the Week 7.

After freeze, any change needs approval from all four leads and a bump of
CONTRACT_VERSION.

Table numbering follows CONTRACTS.md section 2:
    2.1 raw_series          2.6 imputation_output (API-facing)
    2.2 series_hourly       2.8 forecasting_output (API-facing)
    2.3 imputed_hourly      2.9 covariate_output (API-facing)
    2.4 demand_daily        2.10 edit_log
    2.5 forecast            2.11 metrics (+ slices.parquet)

Usage
-----
    from contracts.schemas import validate
    validate("forecast", df)                 # raises on failure, returns coerced df

    python -m contracts.schemas forecast data/runs/<run_id>/forecast.parquet

Requires pandera >= 0.19 and pandas.
"""

from __future__ import annotations

import datetime
import sys

import numpy as np
import pandas as pd

try:  # pandera >= 0.24 exposes the pandas API under pandera.pandas
    import pandera.pandas as pa
except ImportError:  # older pandera
    import pandera as pa

from pandera.engines import pandas_engine

CONTRACT_VERSION = "0.1"

# --------------------------------------------------------------------------
# Constants shared with CONTRACTS.md
# --------------------------------------------------------------------------

SPLITS = ["train", "val", "calib", "test"]
KINDS = ["observed", "recovered"]  # imputation_output.kind (2.6)
FEATURES = ["imputation", "forecasting"]
SLICES = ["all", "low_sales", "high_sales"]
METRICS = [
    "wape",
    "wpe",
    "mase",
    "picp95",
    "mpiw95",
]
EDIT_FIELDS_DEMAND = "demand"
HORIZONS = list(range(1, 8))  # 7-day forecast
SEQ_LEN = 24  # hours per day in the raw sequence columns

# Models whose id contains this token are the non-quantile dataset-paper
# baselines (open decision O2): they may have null quantile/interval columns.
BASELINE_TOKEN = "baseline"

# Covariate column names as they appear in the dataset (CONTRACTS.md 2.1+).
COVARIATES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]

TOL = 1e-6  # numeric tolerance for equality / ordering checks

_UUID_RE = r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_YMD_RE = r"^\d{4}-\d{2}-\d{2}$"


# --------------------------------------------------------------------------
# Column helpers
# --------------------------------------------------------------------------


def _str(nullable: bool = False, required: bool = True, **kw) -> pa.Column:
    return pa.Column(str, nullable=nullable, required=required, coerce=True, **kw)


def _date(nullable: bool = False, required: bool = True) -> pa.Column:
    """datetime.date (no time component, no timezone)."""
    return pa.Column(
        pandas_engine.Date, nullable=nullable, required=required, coerce=True
    )


def _datetime(nullable: bool = False, required: bool = True) -> pa.Column:
    """Naive (timezone-independent) datetime.datetime."""
    return pa.Column(
        "datetime64[ns]", nullable=nullable, required=required, coerce=True
    )


def _time(nullable: bool = False, required: bool = True) -> pa.Column:
    """Naive datetime.time. Stored as Python objects (pandas has no native time dtype)."""
    return pa.Column(
        object,
        checks=pa.Check(
            lambda s: s.map(lambda v: isinstance(v, datetime.time)),
            name="is_datetime_time",
            error="values must be datetime.time",
        ),
        nullable=nullable,
        required=required,
    )


def _float(
    nullable: bool = False, ge: float | None = None, required: bool = True
) -> pa.Column:
    checks = [pa.Check.ge(ge)] if ge is not None else []
    return pa.Column(
        "float64", checks=checks, nullable=nullable, required=required, coerce=True
    )


def _int(
    nullable: bool = False,
    ge: int | None = None,
    le: int | None = None,
    dtype: str = "int64",
    required: bool = True,
) -> pa.Column:
    checks = []
    if ge is not None:
        checks.append(pa.Check.ge(ge))
    if le is not None:
        checks.append(pa.Check.le(le))
    return pa.Column(
        dtype, checks=checks, nullable=nullable, required=required, coerce=True
    )


def _bool(nullable: bool = False) -> pa.Column:
    return pa.Column(bool, nullable=nullable, coerce=True)


def _isin(values: list, nullable: bool = False) -> pa.Column:
    return pa.Column(str, checks=pa.Check.isin(values), nullable=nullable, coerce=True)


def _sequence(length: int, kinds: str, name: str) -> pa.Column:
    """Fixed-length numeric sequence per row (list / ndarray cell).

    `kinds` is a string of numpy dtype kinds that the elements may have,
    e.g. "f" for float64 sequences, "iu" for int32 sequences.
    """

    def fn(s: pd.Series) -> pd.Series:
        def ok(v) -> bool:
            if v is None:
                return False
            try:
                arr = np.asarray(v)
            except Exception:  # noqa: BLE001
                return False
            return arr.ndim == 1 and len(arr) == length and arr.dtype.kind in kinds

        return s.map(ok)

    return pa.Column(
        object,
        checks=pa.Check(
            fn,
            name=f"{name}_is_sequence_of_{length}",
            error=f"{name} must be a length-{length} numeric sequence",
        ),
        nullable=False,
    )


def _id_columns() -> dict[str, pa.Column]:
    """Anonymized store/product codes (int64). Never decoded or joined to outside data."""
    return {
        "store_id": _int(),
        "product_id": _int(),
    }


def _hierarchy_columns() -> dict[str, pa.Column]:
    """City, store and product-category codes (CONTRACTS.md 2.1)."""
    return {
        "city_id": _int(),
        "store_id": _int(),
        "management_group_id": _int(),
        "first_category_id": _int(),
        "second_category_id": _int(),
        "third_category_id": _int(),
        "product_id": _int(),
    }


def _covariate_columns() -> dict[str, pa.Column]:
    return {
        "discount": _float(),
        "holiday_flag": _int(dtype="int32"),
        "activity_flag": _int(dtype="int32"),
        "precpt": _float(),
        "avg_temperature": _float(),
        "avg_humidity": _float(),
        "avg_wind_level": _float(),
    }


def _split_fold(prefix: str) -> dict[str, pa.Column]:
    """`<prefix>_split` and `<prefix>_fold` (fold is null for calib and test)."""
    return {
        f"{prefix}_split": _isin(SPLITS),
        f"{prefix}_fold": _int(nullable=True, ge=0, dtype="UInt32"),
    }


def _quantile_columns() -> dict[str, pa.Column]:
    """Raw model quantiles. Null only for non-quantile baselines (checked per table)."""
    return {
        "q025": _float(nullable=True, ge=0.0),
        "q975": _float(nullable=True, ge=0.0),
    }


def _calibrated_columns() -> dict[str, pa.Column]:
    """Populated after conformalization; null before."""
    return {
        "q025_cal": _float(nullable=True, ge=0.0),
        "q975_cal": _float(nullable=True, ge=0.0),
        "qhat95": _float(nullable=True),  # CQR adjustment may be negative
    }


# --------------------------------------------------------------------------
# Dataframe-level check builders
# --------------------------------------------------------------------------


def _ordered(*cols: str, name: str | None = None) -> pa.Check:
    """cols[0] <= cols[1] <= ... wherever both neighbours are non-null."""

    def fn(df: pd.DataFrame) -> bool:
        for a, b in zip(cols[:-1], cols[1:]):
            if a not in df.columns or b not in df.columns:
                continue
            ok = (df[a] <= df[b] + TOL) | df[a].isna() | df[b].isna()
            if not ok.all():
                return False
        return True

    return pa.Check(
        fn,
        name=name or "ordered(" + " <= ".join(cols) + ")",
        error=f"not ordered: {' <= '.join(cols)}",
    )


def _non_baseline_requires(model_col: str, *cols: str) -> pa.Check:
    """Quantile model rows must have the given columns filled. Baselines may leave them null (O2)."""

    def fn(df: pd.DataFrame) -> bool:
        is_base = df[model_col].str.contains(BASELINE_TOKEN, regex=False)
        for c in cols:
            if c in df.columns and df.loc[~is_base, c].isna().any():
                return False
        return True

    return pa.Check(
        fn,
        name=f"non_baseline_requires({', '.join(cols)})",
        error=f"non-baseline rows must have non-null {cols}",
    )


def _hourly_parts_consistent(df: pd.DataFrame) -> bool:
    """`date` and `time` must be the parts of `datetime`."""
    ts = pd.to_datetime(df["datetime"])
    date_ok = pd.to_datetime(df["date"]) == ts.dt.normalize()
    time_ok = ts.dt.time == df["time"]
    return bool((date_ok & time_ok).all())


_HOURLY_PARTS_CHECK = pa.Check(
    _hourly_parts_consistent,
    name="date_and_time_match_datetime",
    error="date/time columns disagree with datetime",
)


def _target_date_matches(df: pd.DataFrame) -> bool:
    expected = pd.to_datetime(df["origin_date"]) + pd.to_timedelta(
        df["horizon"].astype("int64"), unit="D"
    )
    return bool((pd.to_datetime(df["target_date"]) == expected).all())


_TARGET_DATE_CHECK = pa.Check(
    _target_date_matches,
    name="target_date_is_origin_plus_horizon",
    error="target_date != origin_date + horizon",
)

# --------------------------------------------------------------------------
# 2.1 raw_series
# --------------------------------------------------------------------------

raw_series = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "dt": _str(checks=pa.Check.str_matches(_YMD_RE)),  # string, YYYY-MM-DD
        "sale_amount": _float(ge=0.0),  # normalized units
        "hours_sale": _sequence(SEQ_LEN, "f", "hours_sale"),
        "stock_hour6_22_cnt": _int(ge=0, dtype="int32"),
        "hours_stock_status": _sequence(SEQ_LEN, "iu", "hours_stock_status"),
        **_covariate_columns(),
    },
    unique=["store_id", "product_id", "dt"],
    strict="filter",
    name="raw_series",
    description="One row per store-product-day, with 24-hour sequences. Raw data format, unmodified.",
)

# --------------------------------------------------------------------------
# 2.2 series_hourly
# --------------------------------------------------------------------------

series_hourly = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "date": _date(),
        "time": _time(),
        "datetime": _datetime(),
        # normalized units; censored (not demand) where is_stockout
        "scaled_sales_quantity": _float(ge=0.0),
        "is_stockout": _bool(),
        **_covariate_columns(),
    },
    checks=[_HOURLY_PARTS_CHECK],
    unique=["store_id", "product_id", "datetime"],
    strict="filter",  # drop unknown columns rather than fail; use strict=True after freeze if preferred
    name="series_hourly",
    description="One row per store-product-hour. Stockout rows are censored, not zero demand.",
)

# --------------------------------------------------------------------------
# 2.3 imputed_hourly
# --------------------------------------------------------------------------

imputed_hourly = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "date": _date(),
        "time": _time(),
        "datetime": _datetime(),
        "impute_model_id": _str(),
        "impute_seed": _int(ge=0, dtype="uint32"),
        **_split_fold("impute"),
        # Non-quantile baselines have no interval (open decision O2); every other
        # model must fill both (checked below by _non_baseline_requires).
        **_quantile_columns(),
        "estimated_demand": _float(nullable=False, ge=0.0),
        # True sales for masked in-stock hours (calibration/evaluation only).
        # Null for real stockout hours, where no ground truth exists.
        "actual_demand": _float(nullable=True, ge=0.0),
        **_calibrated_columns(),
    },
    checks=[
        _HOURLY_PARTS_CHECK,
        _ordered(
            "q025", "estimated_demand", "q975", name="imputed_point_inside_raw_interval"
        ),
        _ordered(
            "q025_cal",
            "estimated_demand",
            "q975_cal",
            name="imputed_point_inside_cal_interval",
        ),
        _non_baseline_requires("impute_model_id", "q025", "q975"),
    ],
    unique=[
        "store_id",
        "product_id",
        "datetime",
        "impute_model_id",
        "impute_seed",
        "impute_fold",
    ],
    strict="filter",
    name="imputed_hourly",
    description="Imputed demand for stockout hours, plus masked in-stock hours (non-null actual_demand) in calibration/evaluation runs.",
)

# --------------------------------------------------------------------------
# 2.4 demand_daily
# --------------------------------------------------------------------------


def _demand_total_consistent(df: pd.DataFrame) -> bool:
    return bool(
        np.isclose(
            df["demand_total"],
            df["demand_observed"] + df["demand_recovered"],
            atol=1e-4,
            rtol=1e-6,
        ).all()
    )


demand_daily = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "date": _date(),
        "imputation_run_id": _str(),
        "demand_observed": _float(nullable=False, ge=0.0),
        "demand_recovered": _float(nullable=False, ge=0.0),
        "demand_total": _float(nullable=False, ge=0.0),
        "stock_hour6_22_cnt": _int(ge=0, dtype="int32"),
        **_covariate_columns(),
        **_split_fold("forecast"),
    },
    checks=[
        pa.Check(
            _demand_total_consistent,
            name="demand_total_is_observed_plus_recovered",
            error="demand_total != demand_observed + demand_recovered",
        ),
    ],
    unique=[
        "store_id",
        "product_id",
        "date",
        "imputation_run_id",
        "forecast_split",
        "forecast_fold",
    ],
    strict="filter",
    name="demand_daily",
    description="Observed + recovered daily demand. Forecast input and target. Versioned by imputation_run_id.",
)

# --------------------------------------------------------------------------
# 2.5 forecast
# --------------------------------------------------------------------------

_FORECAST_RUN_KEYS = [
    "store_id",
    "product_id",
    "origin_date",
    "imputation_run_id",
    "forecast_model_id",
    "forecast_seed",
    "forecast_fold",
]


def _complete_horizons(df: pd.DataFrame) -> bool:
    """Every (store, product, origin, imputation run, model, seed, fold) must carry all 7 horizons."""
    want = set(HORIZONS)
    # dropna=False: fold is null for calib/test and those groups must still be checked
    return bool(
        df.groupby(_FORECAST_RUN_KEYS, dropna=False)["horizon"]
        .agg(lambda s: set(int(h) for h in s) == want)
        .all()
    )


def _run_id_matches(df: pd.DataFrame) -> bool:
    """run_id is `<model_id>-<seed>` (CONTRACTS.md section 1)."""
    expected = df["forecast_model_id"] + "-" + df["forecast_seed"].astype(str)
    return bool((df["forecast_run_id"] == expected).all())


forecast = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "date": _date(),
        "imputation_run_id": _str(),
        "forecast_model_id": _str(),
        "forecast_seed": _int(ge=0, dtype="uint32"),
        **_split_fold("forecast"),
        "forecast_run_id": _str(),
        "origin_date": _date(),  # last day of history used
        "horizon": _int(ge=1, le=7, dtype="uint8"),
        "target_date": _date(),
        # Baseline models (O2) may leave quantiles null; others may not (checked below).
        **_quantile_columns(),
        "estimated_demand": _float(nullable=False, ge=0.0),
        # demand_total for target_date, when it exists. Only evaluation code reads this column.
        "actual_demand": _float(nullable=True, ge=0.0),
        **_calibrated_columns(),
    },
    checks=[
        _TARGET_DATE_CHECK,
        pa.Check(
            _complete_horizons,
            name="all_seven_horizons_present",
            error="some forecast origins are missing horizons 1..7",
        ),
        pa.Check(
            _run_id_matches,
            name="run_id_is_model_id_dash_seed",
            error="forecast_run_id != <forecast_model_id>-<forecast_seed>",
        ),
        _ordered(
            "q025", "estimated_demand", "q975", name="forecast_point_inside_raw_interval"
        ),
        _ordered(
            "q025_cal",
            "estimated_demand",
            "q975_cal",
            name="forecast_point_inside_cal_interval",
        ),
        _non_baseline_requires("forecast_model_id", "q025", "q975"),
    ],
    unique=[
        "store_id",
        "product_id",
        "origin_date",
        "horizon",
        "imputation_run_id",
        "forecast_model_id",
        "forecast_seed",
        "forecast_fold",
    ],
    strict="filter",
    name="forecast",
    description="Forecasts with raw and (once conformalized) calibrated quantiles. One row per store-product-forecast-day.",
)

# --------------------------------------------------------------------------
# 2.6 imputation_output  (API-facing)
# --------------------------------------------------------------------------


def _observed_rows_have_no_interval(df: pd.DataFrame) -> bool:
    o = df[df["kind"] == "observed"]
    return bool(o[["q025_cal", "q975_cal"]].isna().all().all())


def _recovered_rows_have_range(df: pd.DataFrame) -> bool:
    """Display rule: recovered estimates are never shown without their range."""
    r = df[df["kind"] == "recovered"]
    return bool(not r[["q025_cal", "q975_cal"]].isna().any().any())


imputation_output = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "date": _date(),
        "time": _time(),
        "datetime": _datetime(),
        "kind": _isin(KINDS),
        "demand": _float(nullable=False, ge=0.0),
        "q025_cal": _float(nullable=True, ge=0.0),
        "q975_cal": _float(nullable=True, ge=0.0),
    },
    checks=[
        _HOURLY_PARTS_CHECK,
        pa.Check(
            _observed_rows_have_no_interval,
            name="observed_rows_have_null_interval",
            error="observed rows must have null q025_cal / q975_cal",
        ),
        pa.Check(
            _recovered_rows_have_range,
            name="recovered_rows_have_interval",
            error="recovered rows must carry a calibrated interval",
        ),
        _ordered("q025_cal", "demand", "q975_cal", name="demand_inside_cal_interval"),
    ],
    unique=["store_id", "product_id", "datetime"],
    strict="filter",
    name="imputation_output",
    description="API-facing hourly imputation series: observed and recovered demand, one row per store-product-hour.",
)

# --------------------------------------------------------------------------
# 2.8 forecasting_output  (API-facing)
# --------------------------------------------------------------------------

forecasting_output = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "origin_date": _date(),
        "horizon": _int(ge=1, le=7, dtype="uint8"),
        "target_date": _date(),
        "date": _date(),
        "demand": _float(nullable=False, ge=0.0),  # forecast, or actual for the origin date
        "q025_cal": _float(nullable=True, ge=0.0),
        "q975_cal": _float(nullable=True, ge=0.0),
    },
    checks=[
        _TARGET_DATE_CHECK,
        _ordered("q025_cal", "demand", "q975_cal", name="demand_inside_cal_interval"),
    ],
    unique=["store_id", "product_id", "origin_date", "horizon", "date"],
    strict="filter",
    name="forecasting_output",
    description="API-facing 7-day daily forecast series, one row per store-product-forecast-day.",
)

# --------------------------------------------------------------------------
# 2.9 covariate_output  (API-facing)
# --------------------------------------------------------------------------

covariate_output = pa.DataFrameSchema(
    columns={
        **_hierarchy_columns(),
        "date": _date(),
        **_covariate_columns(),
    },
    unique=["store_id", "product_id", "date"],
    strict="filter",
    name="covariate_output",
    description="API-facing covariate tab, one row per historical store-product-day.",
)

# --------------------------------------------------------------------------
# 2.10 edit_log  (API-owned, append-only)
# --------------------------------------------------------------------------

edit_log = pa.DataFrameSchema(
    columns={
        "edit_id": _str(checks=pa.Check.str_matches(_UUID_RE)),
        **_id_columns(),
        "date": _date(),
        "field": _str(),  # "demand" or a covariate name
        "old_value": _float(nullable=False),
        "new_value": _float(nullable=False),
        "edited_at": pa.Column("datetime64[ns, UTC]", nullable=False, coerce=True),
        "reason": pa.Column(
            str,
            checks=pa.Check(
                lambda s: s.str.strip().str.len() > 0,
                name="reason_not_blank",
                error="edit reason is required",
            ),
            nullable=False,
            coerce=True,
        ),
        "applied_in_run_id": _str(nullable=True),
    },
    checks=[
        pa.Check(
            lambda df: df["field"].isin([EDIT_FIELDS_DEMAND] + COVARIATES).all(),
            name="field_is_demand_or_covariate",
            error=f"field must be one of {[EDIT_FIELDS_DEMAND] + COVARIATES}",
        ),
        pa.Check(
            lambda df: (df["old_value"] != df["new_value"]).all(),
            name="edit_changes_value",
            error="old_value equals new_value",
        ),
    ],
    unique=["edit_id"],
    strict="filter",
    name="edit_log",
    description="Planner overrides. Append-only. The original dataset is never modified.",
)

# --------------------------------------------------------------------------
# 2.11 metrics  (long format) and slices
# --------------------------------------------------------------------------

metrics = pa.DataFrameSchema(
    columns={
        "feature": _isin(FEATURES),
        "model_id": _str(),
        "fold": _int(ge=0, dtype="int8"),
        "seed": _int(ge=0),
        "split": _isin(SPLITS),
        "slice": _isin(SLICES),
        "metric": _isin(METRICS),
        "value": _float(
            nullable=True
        ),  # null if a metric is undefined (e.g. empty slice)
        "n": _int(ge=0),
        "calibrated": _bool(),
    },
    checks=[
        pa.Check(
            lambda df: (
                df.loc[df["metric"].str.startswith(("picp",)), "value"]
                .dropna()
                .between(0, 1)
                .all()
            ),
            name="picp_in_unit_interval",
            error="PICP must be in [0, 1]",
        ),
        pa.Check(
            lambda df: (
                df.loc[df["metric"].isin(["wape", "mase", "mpiw95"]), "value"].dropna()
                >= 0
            ).all(),
            name="nonnegative_metrics",
            error="WAPE, MASE and MPIW cannot be negative",
        ),
    ],
    unique=[
        "feature",
        "model_id",
        "fold",
        "seed",
        "split",
        "slice",
        "metric",
        "calibrated",
    ],
    strict="filter",
    name="metrics",
    description="Long-format scores so every architecture and slice is comparable.",
)

slices = pa.DataFrameSchema(
    columns={
        **_id_columns(),
        "mean_scaled_demand": _float(nullable=False, ge=0.0),
        "slice": _isin(["low_sales", "high_sales"]),
    },
    checks=[
        pa.Check(
            lambda df: (
                (df["mean_scaled_demand"] < 1) == (df["slice"] == "low_sales")
            ).all(),
            name="slice_matches_threshold",
            error="low_sales must be exactly mean_scaled_demand < 1",
        ),
    ],
    unique=["store_id", "product_id"],
    strict="filter",
    name="slices",
    description="Computed once (slices.parquet). Every consumer uses this membership for low/high-selling reports.",
)

# --------------------------------------------------------------------------
# Registry and entry points
# --------------------------------------------------------------------------

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "raw_series": raw_series,
    "series_hourly": series_hourly,
    "imputed_hourly": imputed_hourly,
    "demand_daily": demand_daily,
    "forecast": forecast,
    "imputation_output": imputation_output,
    "forecasting_output": forecasting_output,
    "covariate_output": covariate_output,
    "edit_log": edit_log,
    "metrics": metrics,
    "slices": slices,
}


def validate(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce `df` against the named schema.

    Collects every failure (lazy=True) so one run reports all problems.
    Raises pandera.errors.SchemaErrors on failure.
    """
    if name not in SCHEMAS:
        raise KeyError(f"unknown table {name!r}; expected one of {sorted(SCHEMAS)}")
    return SCHEMAS[name].validate(df, lazy=True)


def validate_file(name: str, path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return validate(name, df)


def _main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in SCHEMAS:
        print(
            f"usage: python -m contracts.schemas <{'|'.join(sorted(SCHEMAS))}> <parquet path>"
        )
        return 2
    try:
        validate_file(argv[1], argv[2])
    except Exception as e:  # noqa: BLE001
        print(f"INVALID ({argv[1]}, contract v{CONTRACT_VERSION}):\n{e}")
        return 1
    print(f"OK: {argv[2]} matches {argv[1]} (contract v{CONTRACT_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
