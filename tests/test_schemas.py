"""
tests/test_schemas.py

Contract tests (CONTRACTS.md section 8). Four groups:

1. Every synthetic fixture table validates against contracts/schemas.py.
2. Deliberately broken tables are REJECTED, one rule at a time. This is what
   proves the schemas enforce the contract rather than just describing it.
3. Cross-table invariants and CONTRACTS.md <-> schemas.py agreement
   (primary keys), so documentation drift fails CI.
4. contracts/interfaces.py behaviour (signatures, abstractness, helpers).

Run:  pytest tests/test_schemas.py -q
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pandera")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from contracts import interfaces, schemas  # noqa: E402
from fixtures import generate_fixtures as gen  # noqa: E402

try:
    from pandera.errors import SchemaError, SchemaErrors

    SCHEMA_ERRORS: tuple = (SchemaError, SchemaErrors)
except ImportError:  # pragma: no cover - only for stand-ins without pandera.errors
    from pandera.pandas import SchemaErrors

    SCHEMA_ERRORS = (SchemaErrors,)

CONTRACTS_MD = REPO_ROOT / "contracts" / "CONTRACTS.md"
KEYS = gen.KEYS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return gen.build_all(seed=0)


def fresh(tables: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    return tables[name].copy()


# --------------------------------------------------------------------------
# 1. Fixtures validate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(schemas.SCHEMAS))
def test_fixture_validates(tables, name):
    out = schemas.validate(name, fresh(tables, name))
    assert len(out) > 0


@pytest.mark.parametrize("seed", [1, 7])
def test_fixtures_validate_for_other_seeds(seed):
    for name, df in gen.build_all(seed).items():
        schemas.validate(name, df)


def test_fixture_generation_is_deterministic():
    a, b = gen.build_all(0), gen.build_all(0)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].equals(b[k]), f"{k} differs between identical seeds"


def test_every_schema_has_a_fixture_and_vice_versa(tables):
    assert set(tables) == set(schemas.SCHEMAS)


def test_unknown_table_name_is_rejected(tables):
    with pytest.raises(KeyError):
        schemas.validate("not_a_table", tables["slices"])


def test_fixtures_round_trip_through_parquet(tables, tmp_path):
    pytest.importorskip("pyarrow")
    for name, df in tables.items():
        path = tmp_path / f"{name}.parquet"
        schemas.validate(name, df).to_parquet(path, index=False)
        schemas.validate(name, pd.read_parquet(path))


# --------------------------------------------------------------------------
# 2. Broken tables are rejected
# --------------------------------------------------------------------------


def _set(df: pd.DataFrame, i, col: str, value) -> pd.DataFrame:
    df = df.copy()
    df.loc[i, col] = value(df, i) if callable(value) else value
    return df


def _first(df: pd.DataFrame, mask) -> object:
    return df.index[mask][0]


def _dup(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


CASES = [
    # ---- series_hourly
    (
        "series_hourly: hour out of range",
        "series_hourly",
        lambda d: _set(d, d.index[0], "hour", 24),
    ),
    (
        "series_hourly: negative sales",
        "series_hourly",
        lambda d: _set(d, d.index[0], "sales", -1.0),
    ),
    (
        "series_hourly: unknown split",
        "series_hourly",
        lambda d: _set(d, d.index[0], "split", "holdout"),
    ),
    ("series_hourly: duplicate key", "series_hourly", _dup),
    # ---- imputed_hourly
    (
        "imputed_hourly: point above interval",
        "imputed_hourly",
        lambda d: _set(d, d.index[0], "point", lambda x, i: x.loc[i, "q975"] + 1.0),
    ),
    (
        "imputed_hourly: quantile model with null interval",
        "imputed_hourly",
        lambda d: _set(d, _first(d, d["model_id"] == gen.IMP_MODEL), "q975", np.nan),
    ),
    (
        "imputed_hourly: negative actual",
        "imputed_hourly",
        lambda d: _set(d, _first(d, d["actual"].notna()), "actual", -1.0),
    ),
    ("imputed_hourly: duplicate key", "imputed_hourly", _dup),
    # ---- demand_daily
    (
        "demand_daily: total != observed + recovered",
        "demand_daily",
        lambda d: _set(
            d, d.index[0], "demand_total", lambda x, i: x.loc[i, "demand_total"] + 1.0
        ),
    ),
    (
        "demand_daily: recovery on a day with no stockout",
        "demand_daily",
        lambda d: _recovery_without_stockout(d),
    ),
    (
        "demand_daily: n_stockout_hours above 24",
        "demand_daily",
        lambda d: _set(d, d.index[0], "n_stockout_hours", 25),
    ),
    ("demand_daily: duplicate key", "demand_daily", _dup),
    # ---- forecast
    ("forecast: missing a horizon", "forecast", lambda d: d.drop(d.index[0])),
    ("forecast: horizon 8", "forecast", lambda d: _set(d, d.index[0], "horizon", 8)),
    (
        "forecast: target_date != origin + horizon",
        "forecast",
        lambda d: _set(
            d,
            d.index[0],
            "target_date",
            lambda x, i: x.loc[i, "target_date"] + pd.Timedelta(days=1),
        ),
    ),
    (
        "forecast: point above raw interval",
        "forecast",
        lambda d: _set(
            d,
            _first(d, d["model_id"] == gen.FC_QUANTILE),
            "point",
            lambda x, i: x.loc[i, "q975"] + 1.0,
        ),
    ),
    (
        "forecast: quantile model with null interval",
        "forecast",
        lambda d: _set(d, _first(d, d["model_id"] == gen.FC_QUANTILE), "q025", np.nan),
    ),
    (
        "forecast: negative point",
        "forecast",
        lambda d: _set(d, d.index[0], "point", -1.0),
    ),
    ("forecast: duplicate key", "forecast", _dup),
    # ---- forecast_calibrated
    (
        "forecast_calibrated: calibration window ends after origin",
        "forecast_calibrated",
        lambda d: _set(
            d,
            d.index[0],
            "calib_window_end",
            lambda x, i: x.loc[i, "origin_date"] + pd.Timedelta(days=1),
        ),
    ),
    (
        "forecast_calibrated: point below calibrated interval",
        "forecast_calibrated",
        lambda d: _set(d, d.index[0], "lo95", lambda x, i: x.loc[i, "point"] + 1.0),
    ),
    (
        "forecast_calibrated: quantile model missing interval",
        "forecast_calibrated",
        lambda d: _set(d, d.index[0], "hi95", np.nan),
    ),
    (
        "forecast_calibrated: missing a horizon",
        "forecast_calibrated",
        lambda d: d.drop(d.index[0]),
    ),
    # ---- review_output
    (
        "review_output: recovered row without a range",
        "review_output",
        lambda d: _set(d, _first(d, d["kind"] == "recovered"), "lo95", np.nan),
    ),
    (
        "review_output: forecast row without a range",
        "review_output",
        lambda d: _set(d, _first(d, d["kind"] == "forecast"), "hi95", np.nan),
    ),
    (
        "review_output: unknown kind",
        "review_output",
        lambda d: _set(d, d.index[0], "kind", "guess"),
    ),
    (
        "review_output: point outside range",
        "review_output",
        lambda d: _set(
            d,
            _first(d, d["kind"] == "forecast"),
            "point",
            lambda x, i: x.loc[i, "hi95"] + 1.0,
        ),
    ),
    ("review_output: duplicate key", "review_output", _dup),
    # ---- edit_log
    (
        "edit_log: blank reason",
        "edit_log",
        lambda d: _set(d, d.index[0], "reason", "   "),
    ),
    (
        "edit_log: edit that changes nothing",
        "edit_log",
        lambda d: _set(d, d.index[0], "new_value", lambda x, i: x.loc[i, "old_value"]),
    ),
    (
        "edit_log: unknown field",
        "edit_log",
        lambda d: _set(d, d.index[0], "field", "price"),
    ),
    ("edit_log: duplicate edit_id", "edit_log", _dup),
    # ---- metrics
    (
        "metrics: PICP above 1",
        "metrics",
        lambda d: _set(d, _first(d, d["metric"] == "picp95"), "value", 1.5),
    ),
    (
        "metrics: negative WAPE",
        "metrics",
        lambda d: _set(d, _first(d, d["metric"] == "wape"), "value", -1.0),
    ),
    (
        "metrics: unknown slice",
        "metrics",
        lambda d: _set(d, d.index[0], "slice", "medium_sales"),
    ),
    (
        "metrics: unknown metric",
        "metrics",
        lambda d: _set(d, d.index[0], "metric", "rmse"),
    ),
    ("metrics: duplicate key", "metrics", _dup),
    # ---- slices
    (
        "slices: low_sales label above threshold",
        "slices",
        lambda d: _set(
            d, _first(d, d["slice"] == "low_sales"), "mean_scaled_demand", 2.0
        ),
    ),
    ("slices: duplicate series", "slices", _dup),
]


def _recovery_without_stockout(d: pd.DataFrame) -> pd.DataFrame:
    """Keep every other rule satisfied so only the no-recovery rule can fire."""
    d = d.copy()
    i = _first(d, d["n_stockout_hours"] == 0)
    d.loc[i, "demand_recovered"] = 1.0
    d.loc[i, "recovered_hi"] = 1.0
    d.loc[i, "demand_total"] = d.loc[i, "demand_observed"] + 1.0
    return d


@pytest.mark.parametrize("label,name,mutate", CASES, ids=[c[0] for c in CASES])
def test_broken_table_is_rejected(tables, label, name, mutate):
    broken = mutate(fresh(tables, name))
    with pytest.raises(SCHEMA_ERRORS):
        schemas.validate(name, broken)


def test_baseline_forecast_may_omit_interval(tables):
    """Open decision O2: non-quantile baselines have null q025/q975."""
    df = fresh(tables, "forecast")
    base = df[df["model_id"] == gen.FC_BASELINE]
    assert len(base) > 0 and base[["q025", "q975"]].isna().all().all()
    schemas.validate("forecast", df)


def test_baseline_imputation_may_omit_interval(tables):
    """Open decision O2 applies to imputation too, not just forecasting."""
    df = fresh(tables, "imputed_hourly")
    base = df[df["model_id"] == gen.IMP_BASELINE]
    assert len(base) > 0 and base[["q025", "q975"]].isna().all().all()
    assert df.loc[df["model_id"] == gen.IMP_MODEL, ["q025", "q975"]].notna().all().all()
    schemas.validate("imputed_hourly", df)


def test_masked_imputation_rows_are_valid_with_actual(tables):
    df = fresh(tables, "imputed_hourly")
    assert df["actual"].notna().any() and df["actual"].isna().any()
    schemas.validate("imputed_hourly", df)


def test_calibration_window_may_end_on_origin_date(tables):
    """The calibration week is the last week of history, so it ends on the origin."""
    df = fresh(tables, "forecast_calibrated")
    assert (df["calib_window_end"] == df["origin_date"]).all()
    schemas.validate("forecast_calibrated", df)


# --------------------------------------------------------------------------
# 3. Cross-table invariants and contract/document agreement
# --------------------------------------------------------------------------


def test_masked_rows_never_feed_demand_daily(tables):
    imp, daily = tables["imputed_hourly"], tables["demand_daily"]
    real = imp[imp["actual"].isna() & (imp["model_id"] == gen.IMP_MODEL)]
    expected = real.groupby(KEYS)["point"].sum()
    got = daily.set_index(KEYS)["demand_recovered"]
    expected = expected.reindex(got.index).fillna(0.0)
    assert np.allclose(got.to_numpy(), expected.to_numpy())


def test_forecast_actuals_match_demand_daily(tables):
    f, daily = tables["forecast"], tables["demand_daily"]
    total = daily.set_index(KEYS)["demand_total"]
    idx = pd.MultiIndex.from_frame(
        f[["store_id", "product_id", "target_date"]].rename(
            columns={"target_date": "date"}
        )
    )
    assert np.allclose(f["actual"].to_numpy(), total.reindex(idx).to_numpy())


def test_calibrated_forecasts_are_a_subset_of_raw_forecasts(tables):
    key = ["store_id", "product_id", "origin_date", "horizon", "model_id"]
    raw = tables["forecast"].set_index(key)
    cal = tables["forecast_calibrated"].set_index(key)
    assert cal.index.isin(raw.index).all()
    assert np.allclose(cal["point"].to_numpy(), raw.loc[cal.index, "point"].to_numpy())


def test_calibration_never_uses_the_scoring_origin_forecasts(tables):
    """forecast_calibrated rows are the scored (test) origins, not the calibration data."""
    assert set(tables["forecast_calibrated"]["split"]) == {"test"}


def test_review_output_serves_one_model(tables):
    assert tables["review_output"]["model_id"].nunique() == 1


def test_slices_cover_every_series(tables):
    series = tables["demand_daily"][["store_id", "product_id"]].drop_duplicates()
    sl = tables["slices"][["store_id", "product_id"]]
    merged = series.merge(sl, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def _contract_primary_keys() -> dict[str, set[str]]:
    """Parse `### 2.N `table`` headings and their **Primary key:** lines."""
    text = CONTRACTS_MD.read_text()
    out: dict[str, set[str]] = {}
    for m in re.finditer(
        r"^### 2\.\d+ `(\w+)`.*?(?=^### |\Z)", text, flags=re.S | re.M
    ):
        pk = re.search(r"\*\*Primary key:\*\*\s*`([^`]+)`", m.group(0))
        if pk:
            out[m.group(1)] = {c.strip() for c in pk.group(1).split(",")}
    return out


def test_contract_document_exists():
    assert CONTRACTS_MD.exists(), f"expected {CONTRACTS_MD}"


def test_primary_keys_in_contract_match_schemas():
    contract = _contract_primary_keys()
    assert contract, "found no primary keys in CONTRACTS.md; heading format changed?"
    for table, pk in contract.items():
        assert table in schemas.SCHEMAS, (
            f"CONTRACTS.md documents {table}, but schemas.py has no schema"
        )
        assert set(schemas.SCHEMAS[table].unique) == pk, (
            f"{table}: CONTRACTS.md key {sorted(pk)} != schema key {sorted(schemas.SCHEMAS[table].unique)}"
        )


def test_every_contract_table_has_a_primary_key_line():
    text = CONTRACTS_MD.read_text()
    documented = set(re.findall(r"^### 2\.\d+ `(\w+)`", text, flags=re.M))
    # forecast_calibrated says "same key as forecast" and slices is described in 2.8
    assert documented - {"forecast_calibrated"} <= set(_contract_primary_keys())


def test_contract_versions_agree():
    assert schemas.CONTRACT_VERSION == interfaces.CONTRACT_INTERFACES_VERSION


# --------------------------------------------------------------------------
# 4. interfaces.py
# --------------------------------------------------------------------------


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def test_config_is_constructor_only():
    """fit and calibrate_* take data (and a seed) only."""
    assert _params(interfaces.Imputer.fit) == ["self", "train", "seed"]
    assert _params(interfaces.Forecaster.fit) == ["self", "train", "seed"]
    assert _params(interfaces.IntervalWrapper.calibrate_forecast) == [
        "self",
        "calib_preds",
    ]
    assert _params(interfaces.IntervalWrapper.calibrate_imputation) == [
        "self",
        "calib_preds",
    ]
    for cls in (interfaces.Imputer, interfaces.Forecaster, interfaces.IntervalWrapper):
        assert _params(cls.__init__) == ["self", "config"]


@pytest.mark.parametrize(
    "cls", [interfaces.Imputer, interfaces.Forecaster, interfaces.IntervalWrapper]
)
def test_interfaces_are_abstract(cls):
    with pytest.raises(TypeError):
        cls({})


def test_interval_wrapper_must_handle_both_domains():
    class ForecastOnly(interfaces.IntervalWrapper):
        def calibrate_forecast(self, calib_preds): ...
        def apply_forecast(self, preds):
            return preds

    with pytest.raises(TypeError):
        ForecastOnly({})


def test_interval_wrapper_may_decline_a_domain_explicitly():
    class Both(interfaces.IntervalWrapper):
        def calibrate_forecast(self, calib_preds): ...
        def apply_forecast(self, preds):
            return preds

        def calibrate_imputation(self, calib_preds):
            raise NotImplementedError

        def apply_imputation(self, preds):
            raise NotImplementedError

    w = Both({})
    with pytest.raises(NotImplementedError):
        w.apply_imputation(pd.DataFrame())


def test_finalize_quantiles_clips_orders_and_reports():
    df = pd.DataFrame(
        {
            "q025": [1.0, -0.5, 3.0, np.nan],
            "q975": [2.0, 1.0, 2.0, np.nan],
            "point": [1.5, 0.2, 2.5, 4.0],
        }
    )
    out, frac = interfaces.finalize_quantiles(df)
    assert out["q025"].tolist()[:3] == [1.0, 0.0, 2.0]  # clipped, then swapped
    assert out["q975"].tolist()[:3] == [2.0, 1.0, 3.0]
    assert out.loc[3, "point"] == 4.0 and np.isnan(
        out.loc[3, "q025"]
    )  # null interval left alone
    assert frac == pytest.approx(0.5)
    assert df.loc[1, "q025"] == -0.5  # input not modified


def test_finalize_quantiles_is_a_noop_on_valid_fixtures(tables):
    for name in ("forecast", "imputed_hourly"):
        _, frac = interfaces.finalize_quantiles(fresh(tables, name))
        assert frac == 0.0


def test_predict_checked_validates_and_reports_clipping(tables):
    class Dummy(interfaces.Forecaster):
        def fit(self, train, seed): ...
        def predict(self, history, origin_dates=None):
            return fresh(tables, "forecast")

        def save(self, path): ...
        @classmethod
        def load(cls, path):
            return cls({"model_id": "fc-dummy"})

    out, frac = Dummy({"model_id": "fc-dummy"}).predict_checked(pd.DataFrame())
    assert frac == 0.0 and len(out) == len(tables["forecast"])


def test_predict_checked_rejects_a_bad_table(tables):
    class Bad(interfaces.Forecaster):
        def fit(self, train, seed): ...
        def predict(self, history, origin_dates=None):
            return fresh(tables, "forecast").drop(columns=["horizon"])

        def save(self, path): ...
        @classmethod
        def load(cls, path):
            return cls({})

    with pytest.raises((*SCHEMA_ERRORS, KeyError)):
        Bad({"model_id": "fc-bad"}).predict_checked(pd.DataFrame())
