"""
contracts/schemas.py

Enforced version of CONTRACTS.md. If this file and CONTRACTS.md disagree,
fix both in the same PR.

STATUS: DRAFT 0.1, freeze at the Week 7 Midpoint Design Review (Oct 27, 2026).
After freeze, any change needs approval from all four leads and a bump of
CONTRACT_VERSION.

Usage
-----
    from contracts.schemas import validate
    validate("forecast", df)                 # raises on failure, returns coerced df

    python -m contracts.schemas forecast data/runs/<run_id>/forecast.parquet

Requires pandera >= 0.19 and pandas.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

try:  # pandera >= 0.24 exposes the pandas API under pandera.pandas
    import pandera.pandas as pa
except ImportError:  # older pandera
    import pandera as pa

CONTRACT_VERSION = "0.1"

# --------------------------------------------------------------------------
# Constants shared with CONTRACTS.md
# --------------------------------------------------------------------------

SPLITS = ["train", "val", "calib", "test"]
KINDS = ["observed", "recovered", "forecast"]
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

# Models whose id contains this token are the non-quantile dataset-paper
# baselines (open decision O2): they may have null quantile/interval columns.
BASELINE_TOKEN = "baseline"

COVARIATES = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precipitation",
    "temp_avg",
    "humidity",
    "wind_level",
]

TOL = 1e-6  # numeric tolerance for equality / ordering checks


# --------------------------------------------------------------------------
# Column helpers
# --------------------------------------------------------------------------


def _str(nullable: bool = False, required: bool = True, **kw) -> pa.Column:
    return pa.Column(str, nullable=nullable, required=required, coerce=True, **kw)


def _date(nullable: bool = False, required: bool = True) -> pa.Column:
    return pa.Column(
        "datetime64[ns]", nullable=nullable, required=required, coerce=True
    )


def _float(
    nullable: bool = False, ge: float | None = None, required: bool = True
) -> pa.Column:
    checks = [pa.Check.ge(ge)] if ge is not None else []
    return pa.Column(
        float, checks=checks, nullable=nullable, required=required, coerce=True
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


def _id_columns() -> dict[str, pa.Column]:
    """Anonymized codes. Never decoded or joined to outside data."""
    return {
        "store_id": _str(),
        "product_id": _str(),
    }


def _optional_hierarchy() -> dict[str, pa.Column]:
    """City and product-group codes. Present if the pipeline chooses to carry them."""
    return {
        "city_id": _str(nullable=True, required=False),
        "group_id": _str(nullable=True, required=False),
        "cat1_id": _str(nullable=True, required=False),
        "cat2_id": _str(nullable=True, required=False),
        "cat3_id": _str(nullable=True, required=False),
    }


def _covariate_columns(required: bool = True) -> dict[str, pa.Column]:
    return {
        "discount": _float(nullable=True, required=required),
        "holiday_flag": _int(nullable=True, dtype="Int8", required=required),
        "activity_flag": _int(nullable=True, dtype="Int8", required=required),
        "precipitation": _float(nullable=True, required=required),
        "temp_avg": _float(nullable=True, required=required),
        "humidity": _float(nullable=True, required=required),
        "wind_level": _float(nullable=True, required=required),
    }


def _split_fold(required: bool = True) -> dict[str, pa.Column]:
    return {
        "split": _isin(SPLITS),
        "fold": _int(ge=0, dtype="int8", required=required, nullable=False),
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


def _non_baseline_requires(*cols: str) -> pa.Check:
    """Quantile model rows must have the given columns filled. Baselines may leave them null (O2)."""

    def fn(df: pd.DataFrame) -> bool:
        is_base = df["model_id"].str.contains(BASELINE_TOKEN, regex=False)
        for c in cols:
            if c in df.columns and df.loc[~is_base, c].isna().any():
                return False
        return True

    return pa.Check(
        fn,
        name=f"non_baseline_requires({', '.join(cols)})",
        error=f"non-baseline rows must have non-null {cols}",
    )


def _quantile_columns(levels: dict[str, bool]) -> dict[str, pa.Column]:
    """levels: column name -> nullable. All quantile outputs are >= 0 (clipped upstream, O4)."""
    return {c: _float(nullable=n, ge=0.0, required=(not n)) for c, n in levels.items()}


# --------------------------------------------------------------------------
# 2.1 series_hourly
# --------------------------------------------------------------------------

series_hourly = pa.DataFrameSchema(
    columns={
        **_id_columns(),
        **_optional_hierarchy(),
        "date": _date(),
        "hour": _int(ge=0, le=23, dtype="int8"),
        "sales": _float(nullable=False, ge=0.0),  # normalized units, observed
        "is_stockout": _bool(),
        **_covariate_columns(),
        **_split_fold(),
    },
    unique=["store_id", "product_id", "date", "hour", "split", "fold"],
    strict="filter",  # drop unknown columns rather than fail; use strict=True after freeze if preferred
    name="series_hourly",
    description="One row per store-product-hour. Stockout rows are censored, not zero demand.",
)

# --------------------------------------------------------------------------
# 2.2 imputed_hourly
# --------------------------------------------------------------------------

imputed_hourly = pa.DataFrameSchema(
    columns={
        **_id_columns(),
        "date": _date(),
        "hour": _int(ge=0, le=23, dtype="int8"),
        "model_id": _str(),
        "seed": _int(ge=0),
        **_split_fold(),
        **_quantile_columns({"q025": False, "q975": False}),
        "point": _float(nullable=False, ge=0.0),
        # True sales for masked in-stock hours (calibration/evaluation only).
        # Null for real stockout hours, where no ground truth exists.
        "actual": _float(nullable=True, ge=0.0),
        # populated by `make conformal`; null before
        "q025_cal": _float(nullable=True, ge=0.0),
        "q975_cal": _float(nullable=True, ge=0.0),
    },
    checks=[
        _ordered("q025", "point", "q975", name="imputed_point_inside_raw_interval"),
        _ordered(
            "q025_cal", "point", "q975_cal", name="imputed_point_inside_cal_interval"
        ),
        _non_baseline_requires("q025", "q975"),
    ],
    unique=["store_id", "product_id", "date", "hour", "model_id", "seed", "fold"],
    strict="filter",
    name="imputed_hourly",
    description="Imputed demand for stockout hours, plus masked in-stock hours (non-null actual) in calibration/evaluation runs.",
)

# --------------------------------------------------------------------------
# 2.3 demand_daily
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


def _no_recovery_without_stockout(df: pd.DataFrame) -> bool:
    no_so = df["n_stockout_hours"] == 0
    return bool((df.loc[no_so, "demand_recovered"].abs() <= TOL).all())


demand_daily = pa.DataFrameSchema(
    columns={
        **_id_columns(),
        "date": _date(),
        "imputation_run_id": _str(),
        "demand_observed": _float(nullable=False, ge=0.0),
        "demand_recovered": _float(nullable=False, ge=0.0),
        "demand_total": _float(nullable=False, ge=0.0),
        "n_stockout_hours": _int(ge=0, le=24, dtype="int8"),
        # display only; forecast models must not consume these
        "recovered_lo": _float(nullable=True, ge=0.0),
        "recovered_hi": _float(nullable=True, ge=0.0),
        **_covariate_columns(),
        **_split_fold(),
    },
    checks=[
        pa.Check(
            _demand_total_consistent,
            name="demand_total_is_observed_plus_recovered",
            error="demand_total != demand_observed + demand_recovered",
        ),
        pa.Check(
            _no_recovery_without_stockout,
            name="no_recovery_without_stockout",
            error="demand_recovered > 0 on a day with no stockout hours",
        ),
        _ordered(
            "recovered_lo",
            "demand_recovered",
            "recovered_hi",
            name="recovered_inside_bounds",
        ),
    ],
    unique=["store_id", "product_id", "date", "imputation_run_id", "split", "fold"],
    strict="filter",
    name="demand_daily",
    description="Observed + recovered daily demand. Forecast input and target. Versioned by imputation_run_id.",
)

# --------------------------------------------------------------------------
# 2.4 forecast
# --------------------------------------------------------------------------


def _target_date_matches(df: pd.DataFrame) -> bool:
    expected = df["origin_date"] + pd.to_timedelta(df["horizon"], unit="D")
    return bool((df["target_date"] == expected).all())


def _complete_horizons(df: pd.DataFrame) -> bool:
    """Every (store, product, origin, model, seed, fold) must carry all 7 horizons."""
    keys = ["store_id", "product_id", "origin_date", "model_id", "seed", "fold"]
    want = set(HORIZONS)
    return bool(
        df.groupby(keys)["horizon"].agg(lambda s: set(s.tolist()) == want).all()
    )


_FORECAST_COLUMNS = {
    **_id_columns(),
    "origin_date": _date(),
    "horizon": _int(ge=1, le=7, dtype="uint8"),
    "target_date": _date(),
    "model_id": _str(),
    "seed": _int(ge=0),
    **_split_fold(),
    "imputation_run_id": _str(),
    # Baseline models (O2) may leave quantiles null; others may not (checked below).
    "q025": _float(nullable=True, ge=0.0),
    "q975": _float(nullable=True, ge=0.0),
    "point": _float(nullable=False, ge=0.0),
    # Only evaluation code reads this column
    "actual": _float(nullable=True, ge=0.0),
}

forecast = pa.DataFrameSchema(
    columns=_FORECAST_COLUMNS,
    checks=[
        pa.Check(
            _target_date_matches,
            name="target_date_is_origin_plus_horizon",
            error="target_date != origin_date + horizon",
        ),
        pa.Check(
            _complete_horizons,
            name="all_seven_horizons_present",
            error="some forecast origins are missing horizons 1..7",
        ),
        _ordered("q025", "point", "q975", name="forecast_point_inside_raw_interval"),
        _non_baseline_requires("q025", "q975"),
    ],
    unique=[
        "store_id",
        "product_id",
        "origin_date",
        "horizon",
        "model_id",
        "seed",
        "fold",
    ],
    strict="filter",
    name="forecast",
    description="Raw (uncalibrated) 7-day forecasts. One row per store-product-forecast-day.",
)

# --------------------------------------------------------------------------
# 2.5 forecast_calibrated
# --------------------------------------------------------------------------


def _calib_window_before_origin(df: pd.DataFrame) -> bool:
    """Calibration data must not overlap the scoring window (no leakage)."""
    return bool((df["calib_window_end"] < df["origin_date"]).all())


def _build_forecast_calibrated() -> pa.DataFrameSchema:
    cols = dict(_FORECAST_COLUMNS)
    cols.update(
        {
            "lo95": _float(nullable=True, ge=0.0),
            "hi95": _float(nullable=True, ge=0.0),
            "calib_window_start": _date(),
            "calib_window_end": _date(),
            "qhat95": _float(nullable=True),
        }
    )
    return pa.DataFrameSchema(
        columns=cols,
        checks=[
            pa.Check(
                _target_date_matches,
                name="target_date_is_origin_plus_horizon",
                error="target_date != origin_date + horizon",
            ),
            pa.Check(
                _complete_horizons,
                name="all_seven_horizons_present",
                error="some forecast origins are missing horizons 1..7",
            ),
            _ordered("lo95", "point", "hi95", name="point_inside_95_interval"),
            _non_baseline_requires("lo95", "hi95"),
            pa.Check(
                _calib_window_before_origin,
                name="calibration_window_precedes_origin",
                error="calibration window overlaps or follows the forecast origin",
            ),
        ],
        unique=[
            "store_id",
            "product_id",
            "origin_date",
            "horizon",
            "model_id",
            "seed",
            "fold",
        ],
        strict="filter",
        name="forecast_calibrated",
        description="Conformalized forecasts. Same key as `forecast` plus lo/hi bounds.",
    )


forecast_calibrated = _build_forecast_calibrated()

# --------------------------------------------------------------------------
# 2.6 review_output  (the ONLY table the API reads)
# --------------------------------------------------------------------------


def _forecast_rows_complete(df: pd.DataFrame) -> bool:
    f = df[df["kind"] == "forecast"]
    needed = ["lo95", "hi95"]
    return bool(not f[needed].isna().any().any())


def _recovered_rows_have_range(df: pd.DataFrame) -> bool:
    """Display rule: recovered estimates are never shown without their range."""
    r = df[df["kind"] == "recovered"]
    return bool(not r[["lo95", "hi95"]].isna().any().any())


review_output = pa.DataFrameSchema(
    columns={
        **_id_columns(),
        "date": _date(),
        "kind": _isin(KINDS),
        "point": _float(nullable=False, ge=0.0),
        "lo95": _float(nullable=True, ge=0.0),
        "hi95": _float(nullable=True, ge=0.0),
        "n_stockout_hours": _int(ge=0, le=24, dtype="int8"),
        "change_warning": _bool(),
        "review_flag": _bool(),
        "model_id": _str(),
        "run_id": _str(),
        **_covariate_columns(),
    },
    checks=[
        pa.Check(
            _forecast_rows_complete,
            name="forecast_rows_have_interval",
            error="forecast rows need lo95 and hi95",
        ),
        pa.Check(
            _recovered_rows_have_range,
            name="recovered_rows_have_interval",
            error="recovered rows must carry an interval",
        ),
        _ordered("lo95", "point", "hi95", name="point_inside_95_interval"),
    ],
    unique=["store_id", "product_id", "date"],
    strict="filter",
    name="review_output",
    description="Single serving table for the API/dashboard: selected model only.",
)

# --------------------------------------------------------------------------
# 2.7 edit_log  (API-owned, append-only)
# --------------------------------------------------------------------------

edit_log = pa.DataFrameSchema(
    columns={
        "edit_id": _str(),
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
# 2.8 metrics  (long format) and slices
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
    description="Computed once. Every consumer uses this membership for low/high-selling reports.",
)

# --------------------------------------------------------------------------
# Registry and entry points
# --------------------------------------------------------------------------

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "series_hourly": series_hourly,
    "imputed_hourly": imputed_hourly,
    "demand_daily": demand_daily,
    "forecast": forecast,
    "forecast_calibrated": forecast_calibrated,
    "review_output": review_output,
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
