"""DLinear forecaster: faithful to the dataset authors' code, the same windows as the shared pytorch-forecasting
datasets, no test-week leakage, and contract tables for a point model."""

# ruff: noqa: UP008  (the dataset authors' classes below are copied unchanged)

import copy
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.data.splits import final_windows, fold_windows
from uncertainty_aware_planner.data.timeseries_dataset import build_timeseries_datasets
from uncertainty_aware_planner.models.forecasting import dlinear
from uncertainty_aware_planner.models.forecasting.dense_windows import (
    gather,
    index_frame,
)
from uncertainty_aware_planner.models.forecasting.forecast_table import COLUMNS
from uncertainty_aware_planner.pipeline.forecast import run, smoke_config

N_SERIES, HORIZON = 6, 7
CONFIG = "configs/forecasting_models/dlinear_forecasting_baseline.yaml"


# The dataset authors' code, copied unchanged from Dingdong-Inc/frn-50k-baseline @ 8af694a (Apache-2.0):
# demand_forecasting/DLinear/models/dlinear.py and lib/revin.py.
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        padding_size = (self.kernel_size - 1) // 2
        if self.kernel_size % 2 == 0:
            end_padding = padding_size + 1
        else:
            end_padding = padding_size
        front = x[:, 0:1, :].repeat(1, padding_size, 1)
        end = x[:, -1:, :].repeat(1, end_padding, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        kernel_size = configs.moving_avg
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x


def authors_forecast(model, x, revin):
    """exp_main.py: RevIN around the model, then the target (last) channel."""
    if revin:
        norm = RevIN(x.shape[-1], eps=1e-5, affine=False)
        out = norm(model(norm(x, "norm")), "denorm")
    else:
        out = model(x)
    return out[:, :, -1]


@pytest.mark.parametrize("kernel", [28, 25])
@pytest.mark.parametrize("revin", [True, False])
def test_port_matches_authors_model(kernel, revin):
    torch.manual_seed(0)
    theirs = Model(
        SimpleNamespace(
            seq_len=62, pred_len=7, moving_avg=kernel, individual=False, enc_in=7
        )
    )
    ours = dlinear.DLinear(62, 7, kernel, revin=revin)
    ours.seasonal.load_state_dict(theirs.Linear_Seasonal.state_dict())
    ours.trend.load_state_dict(theirs.Linear_Trend.state_dict())
    x = torch.rand(32, 62, 7) * 5
    torch.testing.assert_close(
        ours(x[:, :, -1]), authors_forecast(theirs, x, revin), rtol=1e-5, atol=1e-5
    )


def test_authors_covariate_channels_never_reach_the_target():
    torch.manual_seed(1)
    theirs = Model(
        SimpleNamespace(
            seq_len=62, pred_len=7, moving_avg=28, individual=False, enc_in=7
        )
    )
    x = torch.rand(16, 62, 7)
    changed = x.clone()
    changed[:, :, :-1] = torch.rand(16, 62, 6) * 100
    torch.testing.assert_close(
        authors_forecast(theirs, x, True), authors_forecast(theirs, changed, True)
    )


@pytest.fixture
def dlinear_cfg(tmp_path):
    model_cfg = smoke_config(load_yaml(CONFIG))
    model_cfg["output_dir"] = str(tmp_path / "forecast")
    model_cfg["run_dir"] = str(tmp_path / "runs")
    return model_cfg


def test_learning_rate_schedule_and_history(cfg, demand, dlinear_cfg, tmp_path):
    model_cfg = copy.deepcopy(dlinear_cfg)
    model_cfg["train"] = {**model_cfg["train"], "epochs": 6, "patience": 5}
    train_ds, _, _ = dlinear.build_datasets(demand, cfg, model_cfg, eval_part="calib")
    torch.manual_seed(3)
    model = dlinear.build_model(train_ds, model_cfg)
    ckpt = dlinear.fit(model, train_ds, model_cfg, tmp_path / "run")
    history = json.loads((tmp_path / "run" / "train_history.json").read_text())
    assert [h["lr"] for h in history] == pytest.approx(
        [1e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5]
    )
    state = torch.load(ckpt, weights_only=True)
    assert state["scaler"] == list(train_ds.scaler)


def test_scaler_uses_only_the_train_window(cfg, demand, dlinear_cfg):
    model_cfg = dlinear_cfg
    base, _, _ = dlinear.build_datasets(demand, cfg, model_cfg, eval_part="test")
    final = final_windows(cfg["splits"])
    days = pd.to_datetime(demand["date"]).dt.date
    later = demand.copy()
    later.loc[days > final["train"].end, "demand_total"] = 1e6
    same, _, _ = dlinear.build_datasets(later, cfg, model_cfg, eval_part="test")
    assert same.scaler == base.scaler
    inside = demand.copy()
    inside.loc[days == final["train"].end, "demand_total"] += 50.0
    moved, _, _ = dlinear.build_datasets(inside, cfg, model_cfg, eval_part="test")
    assert moved.scaler != base.scaler


def _shared(ds):
    x, _ = next(iter(ds.to_dataloader(train=False, batch_size=100_000, num_workers=0)))
    index = ds.x_to_index(x)
    index = index.assign(
        store_id=index["store_id"].astype("int64"),
        product_id=index["product_id"].astype("int64"),
        time_idx=index["time_idx"].astype("int64"),
    )
    return index, x["encoder_target"].numpy()


def _dense(data):
    index = index_frame(data.days, data.windows)
    w = data.windows
    return index, gather(data.days.target, w, -w.lookback, w.lookback)


def _sorted(index, values):
    order = np.lexsort((index["time_idx"], index["product_id"], index["store_id"]))
    return index.iloc[order].reset_index(drop=True), values[order]


CASES = [(k, "val", None) for k in range(1, 7)] + [
    (None, part, lookback) for part in ("calib", "test") for lookback in (62, 70)
]


@pytest.mark.parametrize("fold,part,lookback", CASES)
def test_windows_equal_the_shared_datasets(
    cfg, demand, dlinear_cfg, fold, part, lookback
):
    data_cfg = copy.deepcopy(cfg)
    model_cfg = copy.deepcopy(dlinear_cfg)
    if lookback is not None:
        data_cfg["splits"]["forecasting"]["final_encoder_days"] = lookback
        model_cfg["windows"]["final_lookback_days"] = lookback
    shared_train, shared_eval, _ = build_timeseries_datasets(
        demand, data_cfg, fold=fold, eval_part=part
    )
    dense_train, dense_eval, spec = dlinear.build_datasets(
        demand, data_cfg, model_cfg, fold=fold, eval_part=part
    )
    for shared, dense in ((shared_train, dense_train), (shared_eval, dense_eval)):
        s_index, s_values = _sorted(*_shared(shared))
        d_index, d_values = _sorted(*_dense(dense))
        pd.testing.assert_frame_equal(
            s_index[["store_id", "product_id", "time_idx"]],
            d_index[["store_id", "product_id", "time_idx"]],
        )
        np.testing.assert_allclose(s_values, d_values, rtol=1e-6)
    expected_eval = 22 if fold is not None else 1
    assert spec["eval_windows_per_series"] == expected_eval


def test_window_counts(cfg, demand, dlinear_cfg):
    fold_train, fold_eval, _ = dlinear.build_datasets(
        demand, cfg, dlinear_cfg, fold=1, eval_part="val"
    )
    final_train, final_eval, _ = dlinear.build_datasets(
        demand, cfg, dlinear_cfg, eval_part="test"
    )
    assert len(fold_train) == N_SERIES * 1
    assert len(fold_eval) == N_SERIES * 22
    assert len(final_train) == N_SERIES * 15  # 83 train days, 62 + 7 per window
    assert len(final_eval) == N_SERIES


def test_test_week_targets_never_reach_the_model(cfg, demand, dlinear_cfg):
    torch.manual_seed(0)
    first = run(dlinear_cfg, cfg, demand, seed=11, eval_parts=["test"])
    clean = pd.read_parquet(first["outputs"]["test"]["table"])
    corrupted = demand.copy()
    test = final_windows(cfg["splits"])["test"]
    days = pd.to_datetime(corrupted["date"]).dt.date
    corrupted.loc[days >= test.start, "demand_total"] = 1e6
    second = run(dlinear_cfg, cfg, corrupted, seed=11, eval_parts=["test"])
    leaked = pd.read_parquet(second["outputs"]["test"]["table"])
    np.testing.assert_array_equal(
        clean["estimated_demand"].to_numpy(), leaked["estimated_demand"].to_numpy()
    )


def check_point_contract(table: pd.DataFrame, demand: pd.DataFrame, run_id: str):
    assert list(table.columns) == COLUMNS
    assert table["q025"].isna().all() and table["q975"].isna().all()
    assert (table["estimated_demand"] >= 0).all()
    assert table["estimated_demand"].notna().all()
    assert set(table["horizon"]) == set(range(1, HORIZON + 1))
    assert not table.duplicated(
        ["store_id", "product_id", "origin_date", "horizon"]
    ).any()
    assert (table["forecast_run_id"] == run_id).all()
    assert (table["imputation_run_id"] == "imp-fixture-0").all()
    actual = demand.assign(target_date=pd.to_datetime(demand["date"]).dt.date)[
        ["store_id", "product_id", "target_date", "demand_total"]
    ]
    joined = table.merge(actual, on=["store_id", "product_id", "target_date"])
    assert len(joined) == len(table)
    np.testing.assert_allclose(joined["actual_demand"], joined["demand_total"])


def test_fold_run_writes_point_table(cfg, demand, dlinear_cfg):
    log = run(dlinear_cfg, cfg, demand, seed=7, fold=1)
    table_path = Path(log["outputs"]["val"]["table"])
    table = pd.read_parquet(table_path)
    check_point_contract(table, demand, "fc-dlinear-baseline-7")
    val = fold_windows(cfg["splits"])[1]["val"]
    assert len(table) == N_SERIES * (val.days - HORIZON + 1) * HORIZON
    assert table["target_date"].min() == val.start
    assert table["target_date"].max() == val.end
    assert table_path.parent == (
        Path(dlinear_cfg["output_dir"]) / "imp-fixture-0" / "fc-dlinear-baseline-7"
    )
    assert table_path.with_name(table_path.name + ".manifest.json").exists()

    run_dir = (
        Path(dlinear_cfg["run_dir"])
        / "imp-fixture-0"
        / "fc-dlinear-baseline-7"
        / "fold1"
    )
    saved = json.loads((run_dir / "run_log.json").read_text())
    assert saved["dataset_spec"]["lookback_days"] == 21
    assert saved["dataset_spec"]["train_windows_per_series"] == 1
    quantiles = pd.read_parquet(run_dir / "quantiles_val.parquet")
    merged = table.merge(
        quantiles, on=["store_id", "product_id", "origin_date", "horizon"]
    )
    np.testing.assert_allclose(merged["estimated_demand"], merged["q500"])


def test_final_run_forecasts_calib_and_test(cfg, demand, dlinear_cfg):
    log = run(dlinear_cfg, cfg, demand, seed=7, eval_parts=["calib", "test"])
    final = final_windows(cfg["splits"])
    for part in ("calib", "test"):
        table = pd.read_parquet(log["outputs"][part]["table"])
        check_point_contract(table, demand, "fc-dlinear-baseline-7")
        assert len(table) == N_SERIES * HORIZON
        assert (table["origin_date"] == final[part].start - timedelta(days=1)).all()
        assert (table["forecast_split"] == part).all()
    assert log["dataset_spec"]["lookback_days"] == 62
    assert log["dataset_spec"]["train_windows_per_series"] == 15


def test_same_seed_same_forecasts(cfg, demand, dlinear_cfg):
    first = pd.read_parquet(
        run(dlinear_cfg, cfg, demand, seed=5, fold=2)["outputs"]["val"]["table"]
    )
    second = pd.read_parquet(
        run(dlinear_cfg, cfg, demand, seed=5, fold=2)["outputs"]["val"]["table"]
    )
    pd.testing.assert_frame_equal(first, second)


def test_plus_flags_run_end_to_end(cfg, demand, tmp_path):
    model_cfg = smoke_config(
        load_yaml("configs/forecasting_models/dlinear_forecasting_plus.yaml")
    )
    model_cfg["output_dir"] = str(tmp_path / "forecast")
    model_cfg["run_dir"] = str(tmp_path / "runs")
    log = run(model_cfg, cfg, demand, seed=3, eval_parts=["calib", "test"])
    table = pd.read_parquet(log["outputs"]["test"]["table"])
    check_point_contract(table, demand, "fc-dlinear-plus-3")
    assert log["dataset_spec"]["future_covariates"] == [
        "discount",
        "activity_flag",
        "holiday_flag",
        "day_of_week",
    ]
