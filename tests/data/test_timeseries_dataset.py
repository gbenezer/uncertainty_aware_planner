from datetime import date

import pytest
import torch

from tests.data.helpers import fake_imputed_hourly
from uncertainty_aware_planner.data.daily import build_demand_daily
from uncertainty_aware_planner.data.splits import final_windows, fold_windows
from uncertainty_aware_planner.data.timeseries_dataset import build_timeseries_datasets

START = date(2024, 3, 28)


@pytest.fixture(scope="module")
def demand(cube, cfg):
    imputed = fake_imputed_hourly(cube, START, date(2024, 6, 25))
    daily, _ = build_demand_daily(cube, imputed, "imp-test-run-0", cfg["splits"])
    return daily


def _decoder_days(dataset):
    batch, _ = next(iter(dataset.to_dataloader(train=False, batch_size=10_000)))
    return batch["decoder_time_idx"]


def _idx(day: date) -> int:
    return (day - START).days


def test_fold_datasets(demand, cfg):
    train_ds, val_ds, spec = build_timeseries_datasets(demand, cfg, fold=1)
    windows = fold_windows(cfg["splits"])[1]
    assert spec["max_encoder_length"] == 21 and spec["max_prediction_length"] == 7
    # 28 training days hold exactly one 21 + 7 day window per series.
    assert len(train_ds) == 6
    assert _decoder_days(train_ds).max() <= _idx(windows["train"].end)
    days = _decoder_days(val_ds)
    assert days.min() == _idx(windows["val"].start) and days.max() == _idx(
        windows["val"].end
    )
    assert len(val_ds) == 6 * (28 - 7 + 1)


def test_final_calib_and_test_datasets(demand, cfg):
    final = final_windows(cfg["splits"])
    _, calib_ds, spec = build_timeseries_datasets(demand, cfg, eval_part="calib")
    assert spec["max_encoder_length"] == 70
    assert len(calib_ds) == 6
    days = _decoder_days(calib_ds)
    assert days.min() == _idx(final["calib"].start) and days.max() == _idx(
        final["calib"].end
    )

    _, test_ds, _ = build_timeseries_datasets(demand, cfg, eval_part="test")
    assert len(test_ds) == 6
    batch, (target, _) = next(iter(test_ds.to_dataloader(train=False, batch_size=100)))
    assert batch["decoder_time_idx"].min() == _idx(final["test"].start)
    assert torch.all(target == 0)  # test actuals never enter the dataset


def test_missing_recovered_demand_is_reported(cube, cfg):
    imputed = fake_imputed_hourly(cube, START, date(2024, 6, 18))
    daily, _ = build_demand_daily(cube, imputed, "partial", cfg["splits"])
    build_timeseries_datasets(
        daily, cfg, fold=1
    )  # folds end before the calibration week
    with pytest.raises(ValueError, match="must cover every stockout"):
        build_timeseries_datasets(daily, cfg, eval_part="test")


def test_untrained_tft_runs_on_a_batch(demand, cfg):
    from pytorch_forecasting import TemporalFusionTransformer

    train_ds, _, _ = build_timeseries_datasets(demand, cfg, fold=1)
    model = TemporalFusionTransformer.from_dataset(
        train_ds, hidden_size=8, attention_head_size=1
    )
    batch, _ = next(iter(train_ds.to_dataloader(train=True, batch_size=4)))
    with torch.no_grad():
        output = model(batch)
    assert output["prediction"].shape[:2] == (4, 7)
