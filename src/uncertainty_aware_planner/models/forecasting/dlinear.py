"""DLinear forecaster (Feature 2): the baseline is a port of the dataset authors' model; flags add our changes.

Ported from Dingdong-Inc/frn-50k-baseline @ 8af694a (Apache-2.0), demand_forecasting/DLinear, run settings of
train_predict.sh:
    models/dlinear.py   moving_avg (even kernels pad one more day at the end), series_decomp, Model with one
                        Linear(seq_len -> pred_len) shared across channels for the seasonal and the trend parts
    lib/revin.py        RevIN without affine weights, eps 1e-5. train_predict.sh passes `--revin False`, but
                        main.py parses it with argparse type=bool, and bool("False") is True: RevIN was on.
    data_loader.py      one StandardScaler fit on every series' training days; stride-1 windows
    exp/exp_main.py     L1 loss on the target channel, Adam, lr halved after every epoch from the second on
                        (utils/tools.py lradj type1: 1e-3, 1e-3, 5e-4, ...), EarlyStopping on the training loss
                        (patience 5) and the best epoch kept; negative forecasts clipped to 0
The authors' model takes 7 channels (6 covariates and the target), but a channel-shared linear layer never mixes
channels and the loss reads the target only, so the covariates cannot change the forecast: the port is
univariate, which tests/models/forecasting checks against their code.

Flags (dlinear_plus; all off for the baseline):
    mask_stockout_targets  the loss skips target days with a stockout between 6:00 and 22:00, where recorded
                           sales understate demand (evaluation also skips them)
    revin_in_stock_stats   RevIN mean and spread from in-stock lookback days only
    future_covariates      planned inputs of the 7 forecast days (e.g. discount, activity_flag, holiday_flag,
                           day_of_week as one-hot), added through one linear layer in RevIN units
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from uncertainty_aware_planner.data.splits import final_windows, fold_windows
from uncertainty_aware_planner.models.forecasting.dense_windows import (
    DenseDays,
    Windows,
    dense_days,
    eval_windows,
    gather,
    index_frame,
    train_windows,
)

CHECKPOINT = "dlinear.pt"
DAY_OF_WEEK = "day_of_week"


class MovingAverage(nn.Module):
    """Trend of each window: replicate-pad (k-1)//2 days in front (one more at the end for even k), then a
    stride-1 average, so the trend has the input's length."""

    def __init__(self, kernel: int):
        super().__init__()
        self.front = (kernel - 1) // 2
        self.end = self.front + (1 if kernel % 2 == 0 else 0)
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L] -> [B, L]
        padded = torch.cat(
            [x[:, :1].repeat(1, self.front), x, x[:, -1:].repeat(1, self.end)], dim=1
        )
        return self.avg(padded.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        kernel: int,
        revin: bool = True,
        eps: float = 1e-5,
        n_future: int = 0,
    ):
        super().__init__()
        self.decomposition = MovingAverage(kernel)
        self.seasonal = nn.Linear(lookback, horizon)
        self.trend = nn.Linear(lookback, horizon)
        self.revin, self.eps = revin, eps
        self.future = nn.Linear(n_future, 1) if n_future else None

    def forward(
        self,
        x: torch.Tensor,  # [B, L] scaled target history
        weights: torch.Tensor
        | None = None,  # [B, L] days that count for the RevIN statistics
        future: torch.Tensor
        | None = None,  # [B, H, K] planned inputs of the forecast days
    ) -> torch.Tensor:  # [B, H]
        if self.revin:
            if weights is None:
                mean = x.mean(dim=1, keepdim=True)
                var = x.var(dim=1, keepdim=True, unbiased=False)
            else:
                weights = torch.where(
                    weights.sum(dim=1, keepdim=True) > 0,
                    weights,
                    torch.ones_like(weights),
                )
                total = weights.sum(dim=1, keepdim=True)
                mean = (x * weights).sum(dim=1, keepdim=True) / total
                var = ((x - mean) ** 2 * weights).sum(dim=1, keepdim=True) / total
            std = torch.sqrt(var + self.eps)
            x = (x - mean) / std
        trend = self.decomposition(x)
        y = self.seasonal(x - trend) + self.trend(trend)
        if self.future is not None:
            y = y + self.future(future).squeeze(-1)
        if self.revin:
            y = y * std + mean
        return y


@dataclass
class DenseForecastData:
    days: DenseDays
    windows: Windows
    scaler: tuple[float, float]  # target mean and std over every series' training days

    def __len__(self) -> int:
        return len(self.windows)


def _future_names(model_cfg: dict) -> list[str]:
    return list(model_cfg["model"].get("future_covariates") or [])


def build_datasets(
    demand_daily: pd.DataFrame,
    data_cfg: dict,
    model_cfg: dict,
    fold: int | None = None,
    eval_part: str = "val",
) -> tuple[DenseForecastData, DenseForecastData, dict]:
    """(train data, eval data, spec), with the windows of data/timeseries_dataset.py and lookbacks from the
    model config. Targets from the first day of a final eval part on are hidden from the eval data."""
    splits = data_cfg["splits"]
    horizon = splits["forecasting"]["horizon_days"]
    windows_cfg = model_cfg.get("windows", {})
    if fold is not None:
        if eval_part != "val":
            raise ValueError("folds are evaluated on 'val'")
        train_window, eval_window = (
            fold_windows(splits)[fold]["train"],
            fold_windows(splits)[fold]["val"],
        )
        lookback = windows_cfg.get(
            "fold_lookback_days", splits["forecasting"]["fold_encoder_days"]
        )
    else:
        if eval_part not in ("calib", "test"):
            raise ValueError("the final split is evaluated on 'calib' or 'test'")
        final = final_windows(splits)
        train_window, eval_window = final["train"], final[eval_part]
        lookback = windows_cfg.get(
            "final_lookback_days", splits["forecasting"]["final_encoder_days"]
        )

    covariates = [c for c in _future_names(model_cfg) if c != DAY_OF_WEEK]
    days = dense_days(
        demand_daily,
        final_windows(splits)["train"].start,
        target=data_cfg["forecasting"]["target"],
        covariates=covariates,
    )
    span = days.target[:, days.day(train_window.start) : days.day(train_window.end) + 1]
    if np.isnan(span).any():
        raise ValueError(
            f"demand_daily has missing targets in the train window {train_window.start}..{train_window.end}"
        )
    std = float(span.std())
    scaler = (float(span.mean()), std if std > 0 else 1.0)

    train = DenseForecastData(
        days, train_windows(days, train_window, lookback, horizon), scaler
    )
    eval_days = (
        days
        if fold is not None
        else days.hide_targets_from(days.day(eval_window.start))
    )
    evaluation = DenseForecastData(
        eval_days, eval_windows(eval_days, eval_window, lookback, horizon), scaler
    )
    spec = {
        "data_path": "dense_windows",
        "lookback_days": lookback,
        "horizon_days": horizon,
        "train_window": [str(train_window.start), str(train_window.end)],
        "eval_window": [str(eval_window.start), str(eval_window.end)],
        "train_windows_per_series": len(train) // days.n_series,
        "eval_windows_per_series": len(evaluation) // days.n_series,
        "target_scaler": {"mean": scaler[0], "std": scaler[1]},
        "future_covariates": _future_names(model_cfg),
    }
    return train, evaluation, spec


def same_preprocessing(a: DenseForecastData, b: DenseForecastData) -> bool:
    return (
        a.scaler == b.scaler
        and a.windows.lookback == b.windows.lookback
        and a.days.keys.equals(b.days.keys)
    )


def build_model(train_ds: DenseForecastData, cfg: dict) -> DLinear:
    return DLinear(**_model_kwargs(train_ds, cfg))


def _model_kwargs(train_ds: DenseForecastData, cfg: dict) -> dict:
    names = _future_names(cfg)
    n_future = len(names) + (6 if DAY_OF_WEEK in names else 0)  # one-hot: 7 columns
    return {
        "lookback": train_ds.windows.lookback,
        "horizon": train_ds.windows.horizon,
        "kernel": int(cfg["model"]["moving_avg"]),
        "revin": bool(cfg["model"].get("revin", True)),
        "n_future": n_future,
    }


def _inputs(data: DenseForecastData, cfg: dict) -> dict[str, torch.Tensor | None]:
    """Model inputs for every window: scaled lookback, RevIN weights and planned inputs of the forecast days."""
    w, days = data.windows, data.days
    mean, std = data.scaler
    x = (gather(days.target, w, -w.lookback, w.lookback) - mean) / std
    weights = None
    if cfg["model"].get("revin_in_stock_stats"):
        weights = gather(days.in_stock.astype("float64"), w, -w.lookback, w.lookback)
    future = None
    names = _future_names(cfg)
    if names:
        columns = []
        for name in names:
            if name == DAY_OF_WEEK:
                dow = (
                    days.start.weekday()
                    + w.first_day[:, None]
                    + np.arange(w.horizon)[None, :]
                ) % 7
                columns.extend((dow == k).astype("float64") for k in range(7))
            else:
                columns.append(gather(days.covariates[name], w, 0, w.horizon))
        future = np.stack(columns, axis=-1)

    def tensor(a):
        return None if a is None else torch.as_tensor(a, dtype=torch.float32)

    return {"x": tensor(x), "weights": tensor(weights), "future": tensor(future)}


def _subset(inputs: dict, idx: torch.Tensor) -> dict:
    return {k: (None if v is None else v[idx]) for k, v in inputs.items()}


def fit(model: DLinear, train_ds: DenseForecastData, cfg: dict, run_dir) -> Path:
    """Train as the authors do; returns the checkpoint (weights, target scaler, model settings, history)."""
    train = cfg["train"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(train.get("device", "cpu"))
    model.to(device)

    w = train_ds.windows
    mean, std = train_ds.scaler
    inputs = {
        k: (None if v is None else v.to(device))
        for k, v in _inputs(train_ds, cfg).items()
    }
    y = torch.as_tensor(
        (gather(train_ds.days.target, w, 0, w.horizon) - mean) / std,
        dtype=torch.float32,
        device=device,
    )
    mask = None
    if cfg["model"].get("mask_stockout_targets"):
        mask = torch.as_tensor(
            gather(train_ds.days.in_stock.astype("float64"), w, 0, w.horizon),
            dtype=torch.float32,
            device=device,
        )

    # Seeded from torch's global generator, which pipeline.forecast seeds per run.
    generator = torch.Generator().manual_seed(
        int(torch.randint(0, 2**31 - 1, (1,)).item())
    )
    lr0 = float(train["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0)
    n, batch = len(y), int(train["batch_size"])
    n_batches = n // batch if train.get("drop_last", True) else -(-n // batch)
    if (
        n_batches == 0
    ):  # fewer windows than one batch (tiny test data): one batch of everything
        n_batches, batch = 1, n

    best_loss, best_state, worse, history = float("inf"), None, 0, []
    for epoch in range(int(train["epochs"])):
        lr = optimizer.param_groups[0]["lr"]
        order = torch.randperm(n, generator=generator).to(device)
        losses = []
        model.train()
        for b in range(n_batches):
            idx = order[b * batch : (b + 1) * batch]
            pred = model(**_subset(inputs, idx))
            error = (pred - y[idx]).abs()
            if mask is None:
                loss = error.mean()
            else:
                loss = (error * mask[idx]).sum() / mask[idx].sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "lr": lr, "train_loss": epoch_loss})
        # EarlyStopping(patience) on the training loss: keep the best (or equal) epoch.
        if epoch_loss <= best_loss:
            best_loss, best_state, worse = (
                epoch_loss,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            worse += 1
            if worse >= int(train.get("patience", 5)):
                break
        for (
            group
        ) in optimizer.param_groups:  # lradj type1 after epoch e (from 0): lr0 * 0.5**e
            group["lr"] = lr0 * 0.5**epoch

    model.load_state_dict(best_state)
    ckpt = run_dir / CHECKPOINT
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "scaler": [float(mean), float(std)],
            "model": _model_kwargs(train_ds, cfg),
        },
        ckpt,
    )
    (run_dir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return ckpt


def predict_quantiles(
    ckpt_path, eval_ds: DenseForecastData, cfg: dict
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Point forecasts in raw units as an (n, horizon, 1) array (one output, labelled 0.5 in the config)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = DLinear(**state["model"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    if tuple(state["scaler"]) != eval_ds.scaler:
        raise ValueError("the eval data were scaled differently from training")
    mean, std = eval_ds.scaler
    inputs = _inputs(eval_ds, cfg)
    with torch.no_grad():
        chunks = [
            model(**_subset(inputs, torch.arange(i, min(i + 65536, len(eval_ds)))))
            for i in range(0, len(eval_ds), 65536)
        ]
    pred = torch.cat(chunks).numpy().astype("float64") * std + mean
    index = index_frame(eval_ds.days, eval_ds.windows)
    lengths = np.full(len(eval_ds), eval_ds.windows.horizon)
    return pred[..., None], index, lengths
