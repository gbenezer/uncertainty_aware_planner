"""Step 5: training-ready PyPOTS inputs for the imputation models (DLinear, TimesNet).

Layout follows the dataset authors (latent_demand_recovery/exp/data/generate_data.py): windows of whole
days flattened to (days x hours) steps, features [sales, covariates..., hour of day], stockout hours NaN,
covariates divided by their window max + 0.1. Two Feature Card differences: all 24 hours are used (D6),
and the calibration week never enters a training set (D7).

Parts: fold k -> "train", "val"; final -> "train", "calib", "recovery" (imputes real stockouts through the
calibration week; each day is owned by exactly one window, see `keep_start` in the index).
"""

from __future__ import annotations

from datetime import timedelta
from itertools import pairwise

import numpy as np
import pandas as pd

from .cube import Cube
from .masks import masks_to_array
from .splits import Window, final_windows, fold_windows

HOURS = 24


def tile(window: Window, window_days: int) -> tuple[list[int], list[int]]:
    """Cover `window` with the fewest `window_days` windows, the last aligned to its end.

    Returns window start offsets (days from window.start) and, for each, the first day it owns.
    """
    if window.days < window_days:
        raise ValueError(
            f"{window.split} window has {window.days} days, fewer than {window_days}"
        )
    starts = list(range(0, window.days - window_days + 1, window_days))
    if starts[-1] + window_days < window.days:
        starts.append(window.days - window_days)
    keep = [starts[0]] + [max(s, prev + window_days) for prev, s in pairwise(starts)]
    return starts, keep


def part_windows(
    splits_cfg: dict, part: str, fold: int | None
) -> tuple[list[Window], list[Window]]:
    """(input windows, owned sub-windows) for one part; the masks of a part are those of its owned window."""
    window_days = splits_cfg["imputation"]["window_days"]
    if fold is not None:
        windows = fold_windows(splits_cfg)[fold]
        if part not in windows:
            raise ValueError(f"fold parts are 'train' and 'val', not {part!r}")
        window = windows[part]
        if window.days != window_days:
            raise ValueError(
                f"fold {part} has {window.days} days but imputation.window_days is {window_days}"
            )
        return [window], [window]

    final = final_windows(splits_cfg)
    if part == "calib":
        calib = final["calib"]
        drawn = Window(
            "calib", None, calib.end - timedelta(days=window_days - 1), calib.end
        )
        return [drawn], [calib]
    if part == "train":
        span = final["train"]
    elif part == "recovery":
        span = Window("recovery", None, final["train"].start, final["calib"].end)
    else:
        raise ValueError(
            f"final parts are 'train', 'calib' and 'recovery', not {part!r}"
        )
    starts, keep = tile(span, window_days)
    inputs = [
        Window(
            span.split,
            None,
            span.start + timedelta(days=s),
            span.start + timedelta(days=s + window_days - 1),
        )
        for s in starts
    ]
    owned = [
        Window(span.split, None, span.start + timedelta(days=k), w.end)
        for k, w in zip(keep, inputs, strict=True)
    ]
    return inputs, owned


def window_features(
    cube: Cube, window: Window, covariates: list[str], scale_offset: float
) -> np.ndarray:
    """(n_series, days * 24, 2 + n_covariates) float32: sales (NaN at stockouts), covariates, hour."""
    a = cube.day_index(window.start)
    b = a + window.days
    sales = np.where(cube.stockout[:, a:b], np.nan, cube.sales[:, a:b]).astype(
        np.float32
    )
    cov = cube.daily_array(covariates, np.float32)[:, a:b]
    cov = cov / (cov.max(axis=1, keepdims=True) + scale_offset)
    shape = sales.shape + (len(covariates),)
    hour = np.broadcast_to(
        (np.arange(HOURS, dtype=np.float32) / (HOURS - 1))[None, None, :, None],
        sales.shape + (1,),
    )
    features = np.concatenate(
        [sales[..., None], np.broadcast_to(cov[:, :, None, :], shape), hour], axis=-1
    )
    return features.reshape(cube.n_series, window.days * HOURS, -1)


def build_pypots_set(
    cube: Cube,
    cfg: dict,
    part: str,
    fold: int | None = None,
    masks: pd.DataFrame | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return a PyPOTS dataset dict and an index with one row per sample (window x series).

    "val" and "calib" need the shared eval masks: X has the masked hours hidden, X_ori keeps their true
    sales. The index has store_id, product_id, start_date, keep_start (first date the sample owns).
    """
    splits_cfg = cfg["splits"]
    imputation = cfg["imputation"]
    inputs, owned = part_windows(splits_cfg, part, fold)
    with_masks = part in ("val", "calib")
    if with_masks and masks is None:
        raise ValueError(f"part {part!r} needs the eval masks")

    xs, x_oris, index = [], [], []
    for window, own in zip(inputs, owned, strict=True):
        x = window_features(
            cube, window, imputation["covariates"], imputation["covariate_scale_offset"]
        )
        if with_masks:
            hidden = np.zeros((cube.n_series, window.days, HOURS), dtype=bool)
            offset = (own.start - window.start).days
            hidden[:, offset:] = masks_to_array(masks, cube, own)
            x_oris.append(x)
            x = x.copy()
            x[..., 0][hidden.reshape(cube.n_series, -1)] = np.nan
        xs.append(x)
        index.append(
            cube.series[["store_id", "product_id"]].assign(
                start_date=window.start, keep_start=own.start
            )
        )

    data = {"X": np.concatenate(xs)}
    if with_masks:
        data["X_ori"] = np.concatenate(x_oris)
    return data, pd.concat(index, ignore_index=True)
