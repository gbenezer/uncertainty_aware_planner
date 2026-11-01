import numpy as np
import pandas as pd
import pytest

from uncertainty_aware_planner.data.masks import (
    make_eval_masks,
    masks_to_array,
    mimic_missing,
)
from uncertainty_aware_planner.data.splits import final_windows, fold_windows


def _positions(cube, masks):
    series = cube.series_index(masks["store_id"], masks["product_id"])
    days = (
        (masks["datetime"].dt.normalize() - pd.Timestamp(cube.dates[0])).dt.days
    ).to_numpy()
    return series, days, masks["datetime"].dt.hour.to_numpy()


def test_masks_hide_only_in_stock_hours_on_fully_in_stock_days(cube, masks):
    assert len(masks) > 0
    series, days, hours = _positions(cube, masks)
    assert not cube.stockout[series, days, hours].any()
    # The authors' procedure only hides hours on days with no stockout hour at all (all 24 hours, D6).
    assert not cube.stockout[series, days].any()


def test_masks_stay_inside_their_windows(cfg, cube, masks):
    splits_cfg = cfg["splits"]
    assert set(masks["split"]) == {"val", "calib"}
    dates = masks["datetime"].dt.normalize()
    for k, windows in fold_windows(splits_cfg).items():
        rows = dates[masks["fold"].eq(k).fillna(False)]
        assert rows.min() >= pd.Timestamp(
            windows["val"].start
        ) and rows.max() <= pd.Timestamp(windows["val"].end)
    calib = final_windows(splits_cfg)["calib"]
    rows = dates[masks["split"] == "calib"]
    assert len(rows) > 0
    assert rows.min() >= pd.Timestamp(calib.start) and rows.max() <= pd.Timestamp(
        calib.end
    )


def test_masks_are_reproducible_and_seed_dependent(cfg, cube, masks):
    pd.testing.assert_frame_equal(make_eval_masks(cube, cfg["splits"]), masks)
    other = make_eval_masks(cube, cfg["splits"], seed=7)
    assert not other[["store_id", "product_id", "datetime"]].equals(
        masks[["store_id", "product_id", "datetime"]]
    )
    assert (other["mask_seed"] == 7).all()


def test_masks_round_trip_to_arrays(cfg, cube, masks):
    window = fold_windows(cfg["splits"])[1]["val"]
    array = masks_to_array(masks, cube, window)
    assert array.shape == (cube.n_series, 28, 24)
    assert array.sum() == masks["fold"].eq(1).fillna(False).sum()


def test_mimic_missing_only_hides_observed_values():
    rng = np.random.default_rng(0)
    x = rng.random((50, 28, 24))
    x[rng.random(x.shape) < 0.05] = np.nan
    masked, hidden = mimic_missing(x, np.random.default_rng(1), p=0.3)
    assert hidden.any()
    assert not np.isnan(x[hidden]).any()
    assert np.isnan(masked[hidden]).all()
    assert np.array_equal(np.isnan(masked), np.isnan(x) | hidden)
    with pytest.raises(ValueError):
        mimic_missing(x, rng, p=1.0)
