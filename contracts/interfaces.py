"""
contracts/interfaces.py

Abstract classes every architecture and interval method implements.
Described in CONTRACTS.md section 3; table definitions are in section 2 and
enforced by contracts/schemas.py.

STATUS: DRAFT 0.1, freeze at the Week 7 Midpoint Design Review (Oct 27, 2026).
After freeze, any change needs approval from all four leads and a bump of
CONTRACT_VERSION in schemas.py.

Rules (CONTRACTS.md section 3)
------------------------------
- One architecture per file; no architecture imports another's code.
- Configuration is passed once, to the constructor. `fit` and `calibrate_*`
  take data (and a seed) only, so there is a single source of truth for a run.
- Imputers and Forecasters return RAW quantiles only. Calibration happens in
  an IntervalWrapper, so conformal prediction, test-time augmentation, and the
  external interval network are interchangeable.
- `predict` is deterministic given a seed and loaded weights.
- Evaluation takes prediction tables and the actuals; it must never import
  from `models/` or `uq/`.

Importing this module does not require pandera. Validation helpers import
`schemas` lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import pandas as pd

CONTRACT_INTERFACES_VERSION = "0.1"  # keep in step with schemas.CONTRACT_VERSION


# --------------------------------------------------------------------------
# Shared post-processing
# --------------------------------------------------------------------------


def finalize_quantiles(
    df: pd.DataFrame,
    lower: str = "q025",
    upper: str = "q975",
    point: str = "point",
) -> tuple[pd.DataFrame, float]:
    """Apply the contract's post-prediction rules to a table of raw quantiles.

    - Clip every value at 0 (open decision O4: quantile heads can go negative).
    - Enforce lower <= point <= upper (monotone quantiles, CONTRACTS.md section 1).

    Returns the corrected copy and the fraction of rows that were changed, which
    the pipeline records as `clipped_fraction` in the run log. Rows with null
    quantiles (baseline models, open decision O2) are left alone.
    """
    out = df.copy()
    before = out[[c for c in (lower, upper, point) if c in out.columns]].copy()

    for c in before.columns:
        out[c] = out[c].clip(lower=0.0)

    if lower in out.columns and upper in out.columns:
        lo = out[[lower, upper]].min(axis=1, skipna=False)
        hi = out[[lower, upper]].max(axis=1, skipna=False)
        has_interval = out[lower].notna() & out[upper].notna()
        out.loc[has_interval, lower] = lo[has_interval]
        out.loc[has_interval, upper] = hi[has_interval]
        if point in out.columns:
            out.loc[has_interval, point] = out.loc[has_interval, point].clip(
                lower=out.loc[has_interval, lower], upper=out.loc[has_interval, upper]
            )

    changed = (out[before.columns] != before) & before.notna()
    frac = float(changed.any(axis=1).mean()) if len(out) else 0.0
    return out, frac


def _validate(table: str, df: pd.DataFrame) -> pd.DataFrame:
    from . import schemas  # lazy: keeps this module importable without pandera

    return schemas.validate(table, df)


# --------------------------------------------------------------------------
# Imputer (Feature 1)
# --------------------------------------------------------------------------


class Imputer(ABC):
    """Recovers demand hidden by stockouts.

    Input: `series_hourly` rows (CONTRACTS.md 2.1).
    Output: `imputed_hourly` rows (2.2), one per stockout hour.
    """

    def __init__(self, config: dict) -> None:
        """`config` is the resolved model YAML (configs/imputation/*.yaml).

        Required keys: model_id, architecture, quantiles, hyperparameters,
        covariates, lag_features. Everything the model needs besides data and
        the seed comes from here.
        """
        self.config = config

    @property
    def model_id(self) -> str:
        """`<feature>-<architecture>-<variant>`, e.g. `imp-timesnet-quantile`."""
        return self.config["model_id"]

    @abstractmethod
    def fit(self, train: pd.DataFrame, seed: int) -> None:
        """Train on `series_hourly` rows.

        Stockout rows (`is_stockout == True`) must be excluded from training
        targets: they are censored, not ground-truth demand.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return `imputed_hourly` rows (2.2) for the stockout hours in `df`.

        Returns raw quantiles `q025` and `q975` plus `point`. `q025_cal` and
        `q975_cal` stay null; calibration is the IntervalWrapper's job.
        Must be deterministic given the seed and loaded weights.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write weights and anything needed to reproduce predictions."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Imputer":
        """Inverse of `save`."""

    def predict_checked(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """`predict`, then clip/order the quantiles and validate against the schema.

        Returns (validated table, clipped_fraction) for the run log. The
        pipeline should call this instead of `predict` directly.
        """
        out, frac = finalize_quantiles(self.predict(df))
        return _validate("imputed_hourly", out), frac


# --------------------------------------------------------------------------
# Forecaster (Feature 2)
# --------------------------------------------------------------------------


class Forecaster(ABC):
    """Forecasts daily demand 7 days ahead.

    Input: `demand_daily` rows (2.3): observed plus recovered demand.
    Output: `forecast` rows (2.4), all 7 horizons per origin.
    """

    def __init__(self, config: dict) -> None:
        """`config` is the resolved model YAML (configs/forecasting/*.yaml)."""
        self.config = config

    @property
    def model_id(self) -> str:
        """e.g. `fc-tft-quantile`."""
        return self.config["model_id"]

    @abstractmethod
    def fit(self, train: pd.DataFrame, seed: int) -> None:
        """Train on `demand_daily` rows (target: `demand_total`).

        Must not read `recovered_lo` / `recovered_hi`; they are display-only.
        """

    @abstractmethod
    def predict(
        self,
        history: pd.DataFrame,
        origin_dates: Sequence[pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        """Return `forecast` rows (2.4): all 7 horizons per origin.

        `history` is `demand_daily` rows up to and including each origin date.
        `origin_dates` selects the forecast origins; None means the last date
        available per series. Each returned row must carry `origin_date`,
        `horizon` (1..7), `target_date = origin_date + horizon`, raw `q025` and
        `q975`, `point`, and `imputation_run_id` copied from `history`. The
        `actual` column is filled by evaluation code, not here.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write weights and anything needed to reproduce predictions."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Forecaster":
        """Inverse of `save`."""

    def predict_checked(
        self,
        history: pd.DataFrame,
        origin_dates: Sequence[pd.Timestamp] | None = None,
    ) -> tuple[pd.DataFrame, float]:
        """`predict`, then clip/order the quantiles and validate against the schema.

        Returns (validated table, clipped_fraction) for the run log.
        """
        out, frac = finalize_quantiles(self.predict(history, origin_dates))
        return _validate("forecast", out), frac


# --------------------------------------------------------------------------
# IntervalWrapper (uncertainty calibration)
# --------------------------------------------------------------------------


class IntervalWrapper(ABC):
    """Calibrates raw quantile output. Conformal prediction is the priority method.

    Works on predictions only. It never imports or calls a model, so any
    Imputer or Forecaster can be wrapped. Forecast and imputation calibration
    are separate methods with independent state: calibrating one never affects
    the other. A method that cannot handle a domain (a stretch-goal method, for
    example) must raise NotImplementedError explicitly rather than guess.
    """

    def __init__(self, config: dict) -> None:
        """`config` is the resolved configs/conformal.yaml.

        Required keys: alphas, score_function, calib_source.
        """
        self.config = config

    # ---- forecasts ------------------------------------------------------

    @abstractmethod
    def calibrate_forecast(self, calib_preds: pd.DataFrame) -> None:
        """Fit on the held-out calibration week of a `forecast` table.

        `calib_preds` has `split == "calib"` and carries the true values in
        `actual`. Calibration data must not overlap the scoring window; the
        `forecast_calibrated` schema checks that `calib_window_end` is on or
        before every `origin_date`.
        """

    @abstractmethod
    def apply_forecast(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Return `forecast_calibrated` rows (2.5) for a `forecast` table.

        Adds `lo95`, `hi95`, `calib_window_start`, `calib_window_end`, and
        `qhat95`. If a point falls outside its calibrated interval, clamp it
        to the nearest bound and report the count to the run log (2.5).
        """

    def apply_forecast_checked(self, preds: pd.DataFrame) -> pd.DataFrame:
        """`apply_forecast`, then validate against the `forecast_calibrated` schema."""
        return _validate("forecast_calibrated", self.apply_forecast(preds))

    # ---- imputations ----------------------------------------------------

    @abstractmethod
    def calibrate_imputation(self, calib_preds: pd.DataFrame) -> None:
        """Fit on held-out calibration data of an `imputed_hourly` table.

        Imputation has no ground truth during stockouts, so calibration uses
        in-stock hours that were masked and imputed. Those rows carry the true
        sales in `actual` (non-null); rows with null `actual` are real
        stockout hours and must be ignored here.
        """

    @abstractmethod
    def apply_imputation(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Return `imputed_hourly` rows (2.2) with `q025_cal` and `q975_cal` filled."""

    def apply_imputation_checked(self, preds: pd.DataFrame) -> pd.DataFrame:
        """`apply_imputation`, then validate against the `imputed_hourly` schema."""
        return _validate("imputed_hourly", self.apply_imputation(preds))
