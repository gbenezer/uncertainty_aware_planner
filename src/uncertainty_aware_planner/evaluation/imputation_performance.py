# derived from https://github.com/Dingdong-Inc/frn-50k-baseline/blob/main/latent_demand_recovery/exp/app.py
# retrieved 30Oct2026

import numpy as np
import warnings
from uncertainty_aware_planner.utils.constants import WINDOW_DURATION, NUMBER_OPEN_HOURS
warnings.filterwarnings('ignore')


def evaluation_mnar(input_data, imputation_data, valid_idx=None, verbose=True):
    """Day-level WAPE / WPE of imputed hourly sales on artificially censored days.

    Args:
        input_data: dict from load_raw_data_for_imputation (or one entry of
            split_by_window_imputation). X[..., 0] holds the censored sales.
        imputation_data: dict with the same "train_set": {"X"} layout, where
            X[..., 0] holds the fully imputed sales (no NaN).
        valid_idx: boolean (n_windows, n_steps, 1) mask of entries that were
            observed originally but masked by mimic_missing. Defaults to
            input_data["valid_idx"].

    A day is evaluated if mimic_missing hid part of it AND it had no real
    stockout hours. Prediction is the imputed daily total over the open hours;
    truth is the original daily total over the open hours (ts_origin).
    """
    if valid_idx is None:
        valid_idx = input_data["valid_idx"]

    # (n_windows, days per window, open hours per day); works for both the full
    # dataset (n_series * WINDOW_NUMBER windows) and a single split window
    day_shape = (-1, WINDOW_DURATION, NUMBER_OPEN_HOURS)

    censored = input_data["train_set"]["X"][..., 0].reshape(day_shape)
    imputed = imputation_data["train_set"]["X"][..., 0].reshape(day_shape)
    truth = input_data["ts_origin"][..., 0].reshape(day_shape)
    masked = valid_idx[..., 0].reshape(day_shape)

    # NaNs in the input = real stockouts + artificially masked entries.
    # valid_idx marks only the artificial ones, so the difference is real stockouts.
    n_artificial = masked.sum(axis=-1)
    n_real_stockout = np.isnan(censored).sum(axis=-1) - n_artificial

    eval_days = (n_artificial > 0) & (n_real_stockout == 0)

    sales_pred = imputed.sum(axis=-1, dtype=np.float64)
    sales_true = truth.sum(axis=-1, dtype=np.float64)

    if np.isnan(sales_pred[eval_days]).any():
        raise ValueError("imputation_data contains NaN on evaluated days")

    err = (sales_pred - sales_true)[eval_days]
    denom = sales_true[eval_days].sum()

    results = {
        "wape": np.abs(err).sum() / denom,
        "wpe": err.sum() / denom,
        "n_eval_days": int(eval_days.sum()),
    }
    if verbose:
        print("wape", results["wape"])
        print("wpe", results["wpe"])
        print("evaluated days", results["n_eval_days"])
    return results