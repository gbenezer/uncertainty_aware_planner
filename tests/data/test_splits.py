import copy
from datetime import date, timedelta

import pytest

from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.splits import (
    draw_cutoffs,
    feasible_cutoffs,
    final_windows,
    fold_windows,
    splits_table,
    validate_splits,
)


@pytest.fixture
def splits_cfg():
    return load_config()["splits"]


def test_committed_splits_validate_against_the_data_calendar(cube, splits_cfg):
    validate_splits(splits_cfg, cube.dates, cube.dates[-7:])


def test_final_split_matches_the_verified_calendar(splits_cfg):
    final = final_windows(splits_cfg)
    assert (final["train"].start, final["train"].end) == (
        date(2024, 3, 28),
        date(2024, 6, 18),
    )
    assert (final["calib"].start, final["calib"].end) == (
        date(2024, 6, 19),
        date(2024, 6, 25),
    )
    assert (final["test"].start, final["test"].end) == (
        date(2024, 6, 26),
        date(2024, 7, 2),
    )
    assert final["train"].days + final["calib"].days == 90


def test_cutoffs_are_the_recorded_seeded_draw(splits_cfg):
    folds = splits_cfg["folds"]
    assert draw_cutoffs(splits_cfg, 6, folds["cutoff_seed"]) == folds["cutoffs"]
    candidates = feasible_cutoffs(splits_cfg)
    assert (candidates[0], candidates[-1], len(candidates)) == (
        date(2024, 4, 25),
        date(2024, 5, 22),
        28,
    )


def test_folds_are_four_week_windows_that_never_reach_calibration(splits_cfg):
    calib_start = final_windows(splits_cfg)["calib"].start
    folds = fold_windows(splits_cfg)
    assert len(folds) == 6
    for windows in folds.values():
        train, val = windows["train"], windows["val"]
        assert train.days == 28 and val.days == 28
        assert val.start == train.end + timedelta(days=1)
        assert train.start >= date(2024, 3, 28)
        assert val.end < calib_start


def test_splits_table(splits_cfg):
    table = splits_table(splits_cfg)
    assert len(table) == 3 + 6 * 2
    assert table.loc[table["fold"].isna(), "split"].tolist() == [
        "train",
        "calib",
        "test",
    ]
    assert set(table.loc[table["fold"].notna(), "split"]) == {"train", "val"}


def test_a_fold_reaching_calibration_is_rejected(cube, splits_cfg):
    bad = copy.deepcopy(splits_cfg)
    bad["folds"]["cutoffs"][-1] = date(2024, 5, 25)
    with pytest.raises(ValueError, match="reaches the calibration"):
        validate_splits(bad, cube.dates, cube.dates[-7:])
