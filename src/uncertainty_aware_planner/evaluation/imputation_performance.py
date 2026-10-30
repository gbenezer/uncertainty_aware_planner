# derived from https://github.com/Dingdong-Inc/frn-50k-baseline/blob/main/latent_demand_recovery/exp/app.py
# retrieved 30Oct2026

import numpy as np
import warnings
from uncertainty_aware_planner.utils.constants import WINDOW_DURATION, NUMBER_OPEN_HOURS
warnings.filterwarnings('ignore')


def evaluation_mnar(input_data, imputation_data, valid_idx=None, lag=1, verbose=True):
    """Day-level WAPE / WPE / MASE of imputed hourly sales on artificially censored days.

    Evaluation is over (series, day) pairs, not hours. Hourly values are summed
    to daily totals over the open hours before any error is computed, and the
    counts reported below are summed over every store-product series in the
    input, so they are far larger than the number of days in a window.

    Args:
        input_data: dict from load_raw_data_for_imputation (or one entry of
            split_by_window_imputation). Must contain:
                "train_set": {"X"}: X[..., 0] holds the censored sales
                    (NaN at real stockouts and at artificially masked hours).
                "ts_origin": original hourly sales before any masking, shape
                    (n_rows, n_steps, 1), used as ground truth.
                "valid_idx": see the valid_idx argument.
            Each row is one store-product series (or one series-window row
            when given the full, unsplit output). Rows are reshaped to
            (n_rows, WINDOW_DURATION, NUMBER_OPEN_HOURS).
        imputation_data: array (not a dict) of shape (n_rows, n_steps, n_features),
            e.g. results["imputation"] from model.predict. Only
            imputation_data[..., 0] is used: the fully imputed hourly sales,
            which must be non-NaN on evaluated days.
        valid_idx: boolean (n_rows, n_steps, 1) mask of entries that were
            observed originally but masked by mimic_missing. Defaults to
            input_data["valid_idx"].
        lag: lag (in days) of the naive forecast used to scale MASE. 1 is the
            plain naive method, 7 is the weekly seasonal naive method.
        verbose: if True, print the metrics and counts.

    Evaluated days:
        A (series, day) pair is evaluated if mimic_missing hid at least one
        hour of it AND it had no real stockout hours. Real stockouts are
        recovered as (NaNs in the input) - (valid_idx entries). Prediction is
        the imputed daily total over the open hours; truth is the original daily
        total over the open hours (ts_origin). Observed hours are included in
        both totals, so days with partial masking are diluted relative to an
        hour-level evaluation of the masked entries only.

    Metrics:
        wape: sum(|pred - true|) / sum(true) over evaluated days.
        wpe: sum(pred - true) / sum(true) over evaluated days. Negative values
            mean a net underestimate of demand.
        mase: mean absolute daily error on evaluated days divided by a single
            pooled naive scale, mean(|y_t - y_{t-lag}|) over all stockout-free
            day pairs (both days clean) in the input. Stockout days are
            excluded from the scale because their "true" totals are censored.
            Values below 1 mean the imputation beats the naive forecast on
            average. The scale is computed per call, so values are not strictly
            comparable across separate splits unless the same pooled scale is
            used.
        mase_window_median: diagnostic only. Each evaluated day's absolute error
            is divided by its own row's naive scale (from at most
            WINDOW_DURATION - lag clean pairs), and the median is reported.
            The mean of these ratios is heavy-tailed and is deliberately not
            reported. Rows whose scale is at or below 1% of the pooled scale
            (including zero or undefined scale) are excluded from this
            diagnostic only.

    Returns:
        dict with keys "wape", "wpe", "mase", "mase_window_median",
        "n_eval_days" (evaluated (series, day) pairs), and "n_mase_days"
        (pairs used for the per-window diagnostic).
    """
    if valid_idx is None:
        valid_idx = input_data["valid_idx"]

    day_shape = (-1, WINDOW_DURATION, NUMBER_OPEN_HOURS)

    censored = input_data["train_set"]["X"][..., 0].reshape(day_shape)
    imputed = imputation_data[..., 0].reshape(day_shape)
    truth = input_data["ts_origin"][..., 0].reshape(day_shape)
    masked = valid_idx[..., 0].reshape(day_shape)

    n_artificial = masked.sum(axis=-1)
    n_real_stockout = np.isnan(censored).sum(axis=-1) - n_artificial

    eval_days = (n_artificial > 0) & (n_real_stockout == 0)

    sales_pred = imputed.sum(axis=-1, dtype=np.float64)
    sales_true = truth.sum(axis=-1, dtype=np.float64)

    if np.isnan(sales_pred[eval_days]).any():
        raise ValueError("imputation_data contains NaN on evaluated days")

    err = (sales_pred - sales_true)[eval_days]
    denom = sales_true[eval_days].sum()

    # ---- MASE ----
    # naive-forecast errors from stockout-free day pairs only
    clean_day = n_real_stockout == 0                                   # (n_windows, days)
    pair_ok = clean_day[:, lag:] & clean_day[:, :-lag]                 # (n_windows, days - lag)
    naive_err = np.abs(sales_true[:, lag:] - sales_true[:, :-lag])

    # pooled scale: mean naive error over all clean day pairs in the data
    n_pairs_total = pair_ok.sum()
    pooled_scale = (naive_err * pair_ok).sum() / n_pairs_total if n_pairs_total > 0 else np.nan
    if not pooled_scale > 0:
        pooled_scale = np.nan

    abs_err_eval = np.abs(err)                                         # eval_days only
    mase = abs_err_eval.mean() / pooled_scale if abs_err_eval.size else np.nan

    # per-window MASE as a diagnostic only (heavy-tailed, so report the median)
    n_pairs = pair_ok.sum(axis=-1)
    scale_w = np.where(
        n_pairs > 0,
        (naive_err * pair_ok).sum(axis=-1) / np.maximum(n_pairs, 1),
        np.nan,
    )
    scale_w = np.where(scale_w > 0.01 * pooled_scale, scale_w, np.nan)  # floor near-degenerate windows
    scaled = np.abs(sales_pred - sales_true) / scale_w[:, None]
    mase_days = eval_days & ~np.isnan(scale_w)[:, None]
    mase_window_median = np.median(scaled[mase_days]) if mase_days.any() else np.nan

    results = {
        "wape": np.abs(err).sum() / denom,
        "wpe": err.sum() / denom,
        "mase": mase,
        "mase_window_median": mase_window_median,
        "n_eval_days": int(eval_days.sum()),
        "n_mase_days": int(mase_days.sum()),
    }
    if verbose:
        print("wape", results["wape"])
        print("wpe", results["wpe"])
        print(f"mase (lag={lag}, pooled scale)", results["mase"])
        print("median per-window scaled error", results["mase_window_median"])
        print("evaluated days across all store-product series", results["n_eval_days"],
              "| days used for per-window diagnostic across all store-product series", results["n_mase_days"])
    return results