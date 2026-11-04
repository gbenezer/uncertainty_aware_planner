"""Step 4: shared imputation evaluation masks (in-stock hours hidden on purpose, then scored).

Every imputation model is scored on the same hidden hours, so the masks are made once and saved.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .cube import Cube
from .splits import Window, final_windows, fold_windows


def mimic_missing(
    x: np.ndarray,
    rng: np.random.Generator,
    p: float,
    max_missing_patch: int = 7,
    min_missing_patch: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of `mimic_missing` from the dataset authors' baseline
    (frn-50k-baseline, latent_demand_recovery/exp/data/generate_data.py).

    x has shape (batch, days, hours) with NaN at real stockout hours. Only fully observed days are
    hidden: each such day with probability p loses a block at its start or end, and runs of whole days
    are hidden with probability p / 10 per day. Returns the masked copy and a boolean array that is True
    exactly at the newly hidden hours. Uses a numpy Generator instead of the global random state; the
    procedure is otherwise unchanged.
    """
    if not 0 < p < 1:
        raise ValueError("missing rate p must be between 0 and 1")
    batch_size, patch_num, patch_len = x.shape
    patch_time = np.arange(patch_len)[None, None, :]
    non_missing_idx = ~np.isnan(x).any(
        axis=-1, keepdims=True
    )  # (batch, days, 1): fully observed days

    # Continuous whole-day runs, ending the day before i, inside a run of fully observed days.
    run_length = np.zeros((batch_size, 1), dtype=int)
    conti_idx = np.zeros_like(non_missing_idx)
    countdown = np.arange(max_missing_patch, 0, -1)[None, :, None]  # 7, 6, ..., 1
    for i in range(patch_num):
        run_length = np.where(non_missing_idx[:, i], run_length, 0) + non_missing_idx[
            :, i
        ].astype(int)
        if i > max_missing_patch:
            conti_len = rng.integers(
                min_missing_patch, max_missing_patch + 1, size=run_length.shape
            )
            conti_len = np.where(
                (conti_len < run_length) & (rng.random(run_length.shape) < p / 10),
                conti_len,
                0,
            )
            conti_tmp = (countdown <= conti_len[:, None]) & (countdown > 1)
            conti_idx[:, i - max_missing_patch + 1 : i + 1] |= conti_tmp

    # Intra-day blocks: hours from a random start to the end of the day, or from the start of the day.
    intra_rand_idx = (rng.random(non_missing_idx.shape) < p) & non_missing_idx
    start_time = rng.integers(0, int(patch_len * (1 - p)), size=non_missing_idx.shape)
    end_time = patch_len - start_time
    use_tail = rng.random(non_missing_idx.shape) <= 0.5
    intra_missing_idx = np.where(
        use_tail, patch_time >= start_time, patch_time <= end_time
    )
    sample_idx = (intra_rand_idx & intra_missing_idx) | conti_idx

    x_missing = np.where(sample_idx, np.nan, x)
    hidden = ~np.isnan(x) & np.isnan(x_missing)
    return x_missing, hidden


def mask_windows(splits_cfg: dict) -> list[tuple[Window, Window]]:
    """(window the masks are drawn on, part of it whose masks are kept) for each fold's val and for calib.

    Calibration masks are drawn on the imputation window that ends on the last calib day, so that whole-day
    runs can occur as in the other windows; only masks inside the calibration week are kept.
    """
    window_days = splits_cfg["imputation"]["window_days"]
    pairs = []
    for windows in fold_windows(splits_cfg).values():
        pairs.append((windows["val"], windows["val"]))
    calib = final_windows(splits_cfg)["calib"]
    drawn = Window(
        "calib", None, calib.end - timedelta(days=window_days - 1), calib.end
    )
    pairs.append((drawn, calib))
    return pairs


def make_eval_masks(
    cube: Cube, splits_cfg: dict, seed: int | None = None
) -> pd.DataFrame:
    """One row per hidden hour: store_id, product_id, datetime, split, fold, mask_seed."""
    mask_cfg = splits_cfg["imputation"]["masks"]
    seed = mask_cfg["seed"] if seed is None else seed
    frames = []
    for drawn, kept in mask_windows(splits_cfg):
        # One independent, reproducible stream per window: fold k uses k, calibration uses 0.
        rng = np.random.default_rng([seed, kept.fold or 0])
        a, b = cube.day_index(drawn.start), cube.day_index(drawn.end) + 1
        x = np.where(cube.stockout[:, a:b], np.nan, cube.sales[:, a:b])
        _, hidden = mimic_missing(
            x,
            rng,
            mask_cfg["missing_rate"],
            mask_cfg["max_missing_days"],
            mask_cfg["min_missing_days"],
        )
        hidden[:, : cube.day_index(kept.start) - a] = False
        series, day, hour = np.nonzero(hidden)
        timestamps = cube.dates[a + day].astype("datetime64[h]") + hour.astype(
            "timedelta64[h]"
        )
        frames.append(
            pd.DataFrame(
                {
                    "store_id": cube.series["store_id"].to_numpy()[series],
                    "product_id": cube.series["product_id"].to_numpy()[series],
                    "datetime": timestamps.astype("datetime64[s]"),
                    "split": kept.split,
                    "fold": pd.array([kept.fold] * len(series), dtype="UInt8"),
                    "mask_seed": seed,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def masks_to_array(masks: pd.DataFrame, cube: Cube, window: Window) -> np.ndarray:
    """Boolean (n_series, window days, 24) array of the hidden hours of one split/fold inside `window`."""
    same_fold = (
        masks["fold"].isna()
        if window.fold is None
        else masks["fold"].eq(window.fold).fillna(False)
    )
    rows = masks[(masks["split"] == window.split) & same_fold]
    a = cube.day_index(window.start)
    days = (
        rows["datetime"].dt.normalize() - pd.Timestamp(window.start)
    ).dt.days.to_numpy()
    inside = (days >= 0) & (days < window.days)
    rows, days = rows[inside], days[inside]
    array = np.zeros((cube.n_series, window.days, 24), dtype=bool)
    series = cube.series_index(rows["store_id"], rows["product_id"])
    array[series, days, rows["datetime"].dt.hour.to_numpy()] = True
    if (array & cube.stockout[:, a : a + window.days]).any():
        raise ValueError("a mask hides a real stockout hour")
    return array
