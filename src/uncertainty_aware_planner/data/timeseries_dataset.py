"""Step 7: training-ready pytorch-forecasting datasets for the forecasting models (TFT, DLinear).

Field roles follow the dataset authors' TFT setup (demand_forecasting/TFT/configs/tft_config.py); the
target is demand_total (observed plus recovered demand). Scalers and encoders are fit on the training
window only; evaluation datasets reuse them via TimeSeriesDataSet.from_dataset.

    fold k -> eval part "val" (rolling 7-day forecasts across the 4-week val window)
    final  -> eval part "calib" or "test" (one 7-day forecast per series)
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from .splits import Window, final_windows, fold_windows


def model_frame(demand_daily: pd.DataFrame, start_date, ts_cfg: dict) -> pd.DataFrame:
    """Add time_idx (days since start_date) and day_of_week; categoricals become string categories."""
    df = demand_daily.copy()
    dates = pd.to_datetime(df["date"])
    df["time_idx"] = (dates - pd.Timestamp(start_date)).dt.days
    df["day_of_week"] = dates.dt.dayofweek
    for column in dict.fromkeys(
        ts_cfg["static_categoricals"] + ts_cfg["time_varying_known_categoricals"]
    ):
        df[column] = df[column].astype(str).astype("category")
    return df


def dataset_spec(ts_cfg: dict, encoder_days: int, horizon_days: int) -> dict:
    """TimeSeriesDataSet keyword arguments (JSON-serializable, for the run log)."""
    return {
        "time_idx": "time_idx",
        "target": ts_cfg["target"],
        "group_ids": ts_cfg["group_ids"],
        "min_encoder_length": encoder_days,
        "max_encoder_length": encoder_days,
        "min_prediction_length": horizon_days,
        "max_prediction_length": horizon_days,
        "static_categoricals": ts_cfg["static_categoricals"],
        "time_varying_known_categoricals": ts_cfg["time_varying_known_categoricals"],
        "time_varying_known_reals": ts_cfg["time_varying_known_reals"],
        "time_varying_unknown_reals": [ts_cfg["target"]],
        "add_relative_time_idx": True,
        "add_encoder_length": True,
        "allow_missing_timesteps": True,
        "target_normalizer": ts_cfg["target_normalizer"],
    }


def _windows(
    splits_cfg: dict, fold: int | None, eval_part: str
) -> tuple[Window, Window, int]:
    forecasting = splits_cfg["forecasting"]
    if fold is not None:
        if eval_part != "val":
            raise ValueError("folds are evaluated on 'val'")
        windows = fold_windows(splits_cfg)[fold]
        return windows["train"], windows["val"], forecasting["fold_encoder_days"]
    if eval_part not in ("calib", "test"):
        raise ValueError("the final split is evaluated on 'calib' or 'test'")
    final = final_windows(splits_cfg)
    return final["train"], final[eval_part], forecasting["final_encoder_days"]


def _between(df: pd.DataFrame, start, end) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    return df[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def build_timeseries_datasets(
    demand_daily: pd.DataFrame,
    cfg: dict,
    fold: int | None = None,
    eval_part: str = "val",
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, dict]:
    """Return (train dataset, eval dataset, dataset spec).

    Test-week targets are replaced by 0 before they enter the dataset (as the dataset authors do), so test
    actuals are never visible to a model; evaluation reads them from demand_daily.
    """
    splits_cfg, ts_cfg = cfg["splits"], cfg["forecasting"]
    train_window, eval_window, encoder_days = _windows(splits_cfg, fold, eval_part)
    spec = dataset_spec(ts_cfg, encoder_days, splits_cfg["forecasting"]["horizon_days"])
    target = ts_cfg["target"]
    start_date = final_windows(splits_cfg)["train"].start
    frame = model_frame(demand_daily, start_date, ts_cfg)

    train_df = _between(frame, train_window.start, train_window.end)
    eval_df = _between(
        frame, eval_window.start - timedelta(days=encoder_days), eval_window.end
    ).copy()
    if eval_part == "test":
        eval_df.loc[
            pd.to_datetime(eval_df["date"]) >= pd.Timestamp(eval_window.start), target
        ] = 0.0
    for name, df in (("train", train_df), (eval_part, eval_df)):
        missing = int(df[target].isna().sum())
        if missing:
            raise ValueError(
                f"{missing} {name} rows have no {target}; the imputation run must cover every stockout "
                f"from {train_window.start} to {eval_window.end}"
            )

    train_ds = TimeSeriesDataSet(train_df, **spec)
    first_eval_idx = (pd.Timestamp(eval_window.start) - pd.Timestamp(start_date)).days
    eval_ds = TimeSeriesDataSet.from_dataset(
        train_ds, eval_df, stop_randomization=True, min_prediction_idx=first_eval_idx
    )
    return train_ds, eval_ds, spec
