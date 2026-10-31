"""Step 3: time-based splits from configs/splits.yaml.

Final split (dataset authors' setup, with our calibration week held out): train, calib, test.
Rolling-origin folds (Feature Cards): 4 weeks train, then 4 weeks val, at 6 fixed cut-off dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Window:
    split: str  # one of train, val, calib, test
    fold: int | None  # None for the final split
    start: date
    end: date  # inclusive

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


def final_windows(splits_cfg: dict) -> dict[str, Window]:
    return {
        name: Window(name, None, *splits_cfg["final"][name])
        for name in ("train", "calib", "test")
    }


def fold_windows(splits_cfg: dict) -> dict[int, dict[str, Window]]:
    folds = splits_cfg["folds"]
    windows = {}
    for k, cutoff in enumerate(folds["cutoffs"], start=1):
        train_start = cutoff - timedelta(days=folds["train_days"])
        val_end = cutoff + timedelta(days=folds["val_days"] - 1)
        windows[k] = {
            "train": Window("train", k, train_start, cutoff - timedelta(days=1)),
            "val": Window("val", k, cutoff, val_end),
        }
    return windows


def splits_table(splits_cfg: dict) -> pd.DataFrame:
    """One row per (fold, split) with inclusive start and end dates; fold is null for the final split."""
    rows = list(final_windows(splits_cfg).values())
    for windows in fold_windows(splits_cfg).values():
        rows.extend(windows.values())
    table = pd.DataFrame(
        {
            "fold": pd.array([w.fold for w in rows], dtype="UInt8"),
            "split": [w.split for w in rows],
            "start_date": [w.start for w in rows],
            "end_date": [w.end for w in rows],
        }
    )
    return table


def feasible_cutoffs(splits_cfg: dict) -> list[date]:
    """Cut-offs (first val day) whose train and val windows fit before the calibration week."""
    folds = splits_cfg["folds"]
    final = final_windows(splits_cfg)
    first = final["train"].start + timedelta(days=folds["train_days"])
    last = final["train"].end - timedelta(days=folds["val_days"] - 1)
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def draw_cutoffs(splits_cfg: dict, n: int, seed: int) -> list[date]:
    """Draw n distinct cut-offs once; the result is committed to splits.yaml as explicit dates."""
    candidates = feasible_cutoffs(splits_cfg)
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(candidates), size=n, replace=False)
    return sorted(candidates[i] for i in picked)


def validate_splits(
    splits_cfg: dict, data_dates: np.ndarray, test_dates: np.ndarray
) -> None:
    """Raise ValueError if the splits overlap, leave the data range, or let a fold touch calib/test."""
    errors = []
    final = final_windows(splits_cfg)
    first_day = pd.Timestamp(data_dates[0]).date()
    last_day = pd.Timestamp(data_dates[-1]).date()

    for prev, nxt in (("train", "calib"), ("calib", "test")):
        if final[nxt].start != final[prev].end + timedelta(days=1):
            errors.append(f"final {nxt} must start the day after final {prev} ends")
    if final["train"].start != first_day:
        errors.append(f"final train must start on the first data day {first_day}")
    if final["test"].end != last_day:
        errors.append(f"final test must end on the last data day {last_day}")
    expected_test = [pd.Timestamp(d).date() for d in test_dates]
    if [final["test"].start, final["test"].end] != [
        expected_test[0],
        expected_test[-1],
    ]:
        errors.append("final test must equal the dates of the official eval split")

    folds_cfg = splits_cfg["folds"]
    if len(set(folds_cfg["cutoffs"])) != len(folds_cfg["cutoffs"]):
        errors.append("fold cut-offs must be distinct")
    for k, windows in fold_windows(splits_cfg).items():
        if windows["train"].start < first_day:
            errors.append(f"fold {k} train starts before the data")
        if windows["val"].end >= final["calib"].start:
            errors.append(f"fold {k} val reaches the calibration or test dates")

    if errors:
        raise ValueError("invalid splits.yaml: " + "; ".join(errors))


def final_split_labels(splits_cfg: dict, dates: pd.Series) -> pd.Series:
    """Label each date with its final split name (train, calib, test)."""
    days = pd.to_datetime(dates)
    labels = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name, window in final_windows(splits_cfg).items():
        labels[
            (days >= pd.Timestamp(window.start)) & (days <= pd.Timestamp(window.end))
        ] = name
    if labels.isna().any():
        raise ValueError("some dates fall outside the final split")
    return labels
