from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from uncertainty_aware_planner.data.pypots_arrays import build_pypots_set, tile
from uncertainty_aware_planner.data.splits import Window, final_windows, fold_windows

STEPS, FEATURES = (
    28 * 24,
    6,
)  # 28-day windows, all 24 hours; sales + 4 covariates + hour


def test_tile_covers_each_day_exactly_once():
    span = Window("train", None, date(2024, 3, 28), date(2024, 6, 18))  # 83 days
    starts, keep = tile(span, 28)
    assert starts == [0, 28, 55] and keep == [0, 28, 56]
    owned = np.concatenate([np.arange(k, s + 28) for s, k in zip(starts, keep)])
    assert np.array_equal(owned, np.arange(83))
    with pytest.raises(ValueError):
        tile(Window("train", None, date(2024, 3, 28), date(2024, 4, 3)), 28)


def test_fold_train_set(cube, cfg):
    data, index = build_pypots_set(cube, cfg, "train", fold=1)
    assert set(data) == {"X"}
    assert data["X"].shape == (cube.n_series, STEPS, FEATURES)
    assert data["X"].dtype == np.float32
    train = fold_windows(cfg["splits"])[1]["train"]
    assert (index["start_date"] == train.start).all()
    # NaN exactly at real stockout hours in the sales channel, nowhere else.
    a = cube.day_index(train.start)
    stockout = cube.stockout[:, a : a + 28].reshape(cube.n_series, -1)
    assert np.array_equal(np.isnan(data["X"][..., 0]), stockout)
    assert not np.isnan(data["X"][..., 1:]).any()
    assert np.allclose(data["X"][0, :24, -1], np.arange(24) / 23)


def test_val_set_hides_exactly_the_shared_masks(cube, cfg, masks):
    data, _ = build_pypots_set(cube, cfg, "val", fold=2, masks=masks)
    x, x_ori = data["X"][..., 0], data["X_ori"][..., 0]
    hidden = np.isnan(x) & ~np.isnan(x_ori)
    assert hidden.sum() == masks["fold"].eq(2).fillna(False).sum()
    assert np.array_equal(np.isnan(x_ori) | hidden, np.isnan(x))


def test_final_train_never_sees_the_calibration_week(cube, cfg):
    data, index = build_pypots_set(cube, cfg, "train")
    assert data["X"].shape == (3 * cube.n_series, STEPS, FEATURES)
    calib_start = final_windows(cfg["splits"])["calib"].start
    last_day = pd.to_datetime(index["start_date"]) + pd.Timedelta(days=27)
    assert (last_day < pd.Timestamp(calib_start)).all()


def test_calib_set_masks_only_the_calibration_week(cube, cfg, masks):
    data, index = build_pypots_set(cube, cfg, "calib", masks=masks)
    calib = final_windows(cfg["splits"])["calib"]
    assert (index["keep_start"] == calib.start).all()
    hidden = (np.isnan(data["X"][..., 0]) & ~np.isnan(data["X_ori"][..., 0])).reshape(
        cube.n_series, 28, 24
    )
    assert hidden.sum() == (masks["split"] == "calib").sum() > 0
    assert not hidden[:, :21].any()


def test_recovery_windows_own_every_day_through_calibration_once(cube, cfg):
    data, index = build_pypots_set(cube, cfg, "recovery")
    assert data["X"].shape == (4 * cube.n_series, STEPS, FEATURES)
    final = final_windows(cfg["splits"])
    first = index[index["store_id"].eq(0) & index["product_id"].eq(100)]
    owned = []
    for start, keep in zip(
        pd.to_datetime(first["start_date"]), pd.to_datetime(first["keep_start"])
    ):
        owned.extend(pd.date_range(keep, start + timedelta(days=27)))
    assert owned == list(pd.date_range(final["train"].start, final["calib"].end))


def test_val_and_calib_need_masks(cube, cfg):
    with pytest.raises(ValueError, match="needs the eval masks"):
        build_pypots_set(cube, cfg, "val", fold=1)


def test_untrained_pypots_dlinear_accepts_the_arrays(cube, cfg):
    from pypots.imputation import DLinear

    data, _ = build_pypots_set(cube, cfg, "train", fold=1)
    model = DLinear(
        n_steps=STEPS,
        n_features=FEATURES,
        moving_avg_window_size=25,
        d_model=64,
        epochs=1,
        verbose=False,
    )
    result = model.predict({"X": data["X"]})
    assert result["imputation"].shape == data["X"].shape
