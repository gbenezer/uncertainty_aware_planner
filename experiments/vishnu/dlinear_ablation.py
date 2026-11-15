"""Choose the improved DLinear's settings on validation data only (never the calibration or test weeks).

Each variant of configs/forecasting_models/dlinear_forecasting_baseline.yaml is trained per seed on
    the inner week: train 03-28..06-11 (62-day lookback), forecast 06-12..06-18 from one origin
    the 6 team folds: 28 train days (21-day lookback), rolling origins across the 4 val weeks
and scored like the paper (raw sales, stockout-free days only, WAPE per origin and target date, then the
mean). Differences against the baseline are paired by seed and fold. Nothing is written to the forecast
folders; one CSV of scores and a markdown table go to --out.

    python experiments/vishnu/dlinear_ablation.py \
        --demand-daily data/processed/demand_daily/imp-none-censored.parquet --out results/forecasting
"""

from __future__ import annotations

import argparse
import copy
import time
from datetime import date

import pandas as pd
import torch

from uncertainty_aware_planner.data.config import load_config, load_yaml, resolve
from uncertainty_aware_planner.data.splits import final_windows
from uncertainty_aware_planner.models.forecasting import dlinear
from uncertainty_aware_planner.models.forecasting.forecast_table import long_quantiles
from uncertainty_aware_planner.utils.util_functions import set_seed

CONFIG = "configs/forecasting_models/dlinear_forecasting_baseline.yaml"
PLANNED = ["discount", "activity_flag", "holiday_flag", "day_of_week"]
VARIANTS = {  # model-section overrides; "lookback" sets windows.final_lookback_days
    "baseline": {},
    "mask": {"mask_stockout_targets": True},
    "instock_revin": {"revin_in_stock_stats": True},
    "planned": {"future_covariates": PLANNED},
    "planned_no_dow": {"future_covariates": PLANNED[:3]},
    "mask+planned": {"mask_stockout_targets": True, "future_covariates": PLANNED},
    "all": {
        "mask_stockout_targets": True,
        "revin_in_stock_stats": True,
        "future_covariates": PLANNED,
    },
    "lookback28": {"lookback": 28},
    "lookback28+all": {
        "lookback": 28,
        "mask_stockout_targets": True,
        "revin_in_stock_stats": True,
        "future_covariates": PLANNED,
    },
}
INNER_TRAIN_END, INNER_WEEK = date(2024, 6, 11), (date(2024, 6, 12), date(2024, 6, 18))


def inner_week_config(data_cfg: dict) -> dict:
    """Final split moved one week back: the 'calib' part is the inner validation week."""
    cfg = copy.deepcopy(data_cfg)
    cfg["splits"]["final"]["train"] = [
        cfg["splits"]["final"]["train"][0],
        INNER_TRAIN_END,
    ]
    cfg["splits"]["final"]["calib"] = list(INNER_WEEK)
    return cfg


def paper_wape_wpe(rows: pd.DataFrame, truth: pd.DataFrame) -> tuple[float, float]:
    """Mean over (origin, target date) cells of pooled WAPE / WPE on stockout-free days (raw sales)."""
    scored = rows.merge(truth, on=["store_id", "product_id", "target_date"])
    scored = scored[scored["stock_hour6_22_cnt"] == 0]
    error = scored["q500"] - scored["actual"]
    cells = scored.assign(abs_error=error.abs(), error=error).groupby(
        ["origin_date", "target_date"]
    )
    sums = cells[["abs_error", "error", "actual"]].sum()
    sums = sums[sums["actual"] > 0]
    return float((sums["abs_error"] / sums["actual"]).mean()), float(
        (sums["error"] / sums["actual"]).mean()
    )


def evaluate_variant(model_cfg, data, start_date, truth, seed):
    train_ds, eval_ds = data
    set_seed(seed)
    model = dlinear.build_model(train_ds, model_cfg)
    ckpt = dlinear.fit(
        model, train_ds, model_cfg, resolve("data/runs/dlinear_ablation")
    )
    q, index, lengths = dlinear.predict_quantiles(ckpt, eval_ds, model_cfg)
    rows, _ = long_quantiles(q, index, lengths, [0.5], start_date)
    return paper_wape_wpe(rows, truth)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demand-daily", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027])
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--out", default="results/forecasting")
    args = parser.parse_args()

    base_cfg = load_yaml(CONFIG)
    data_cfg = load_config(base_cfg["data_config"])
    demand = pd.read_parquet(resolve(args.demand_daily))
    truth = demand.assign(target_date=pd.to_datetime(demand["date"]).dt.date)[
        ["store_id", "product_id", "target_date", "demand_total", "stock_hour6_22_cnt"]
    ].rename(columns={"demand_total": "actual"})
    start_date = final_windows(data_cfg["splits"])["train"].start

    # Data once per (protocol, lookback), with every planned input loaded (variants read only what they use).
    superset = copy.deepcopy(base_cfg)
    superset["model"]["future_covariates"] = PLANNED
    protocols = {"inner_week": (inner_week_config(data_cfg), None, "calib")}
    protocols.update({f"fold{k}": (data_cfg, k, "val") for k in args.folds})
    datasets = {}

    def data_for(name: str, lookback: int):
        if (name, lookback) not in datasets:
            cfg, fold, part = protocols[name]
            wanted = copy.deepcopy(superset)
            wanted["windows"]["final_lookback_days"] = lookback
            train_ds, eval_ds, _ = dlinear.build_datasets(
                demand, cfg, wanted, fold=fold, eval_part=part
            )
            datasets[(name, lookback)] = (train_ds, eval_ds)
        return datasets[(name, lookback)]

    results = []
    for variant in args.variants:
        overrides = dict(VARIANTS[variant])
        lookback = overrides.pop("lookback", base_cfg["windows"]["final_lookback_days"])
        model_cfg = copy.deepcopy(base_cfg)
        model_cfg["model"].update(overrides)
        model_cfg["windows"]["final_lookback_days"] = lookback
        for seed in args.seeds:
            for name in protocols:
                started = time.time()
                wape, wpe = evaluate_variant(
                    model_cfg, data_for(name, lookback), start_date, truth, seed
                )
                results.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "protocol": name,
                        "wape": wape,
                        "wpe": wpe,
                        "seconds": round(time.time() - started, 1),
                    }
                )
                print(
                    f"{variant:16s} seed {seed} {name:10s} WAPE {wape:.4f} WPE {wpe:+.4f}",
                    flush=True,
                )

    scores = pd.DataFrame(results)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "dlinear_ablation.csv", index=False)

    base = scores[scores["variant"] == "baseline"].set_index(["seed", "protocol"])
    lines = [
        (
            "| variant | inner-week WAPE (mean ± SD over seeds) | inner-week WPE | inner-week ΔWAPE vs "
            "baseline (paired by seed) | fold WAPE (mean ± SD) | fold ΔWAPE vs baseline (paired) |"
        ),
        "|---|---|---|---|---|---|",
    ]
    for variant, group in scores.groupby("variant", sort=False):
        g = group.set_index(["seed", "protocol"])
        delta = (g["wape"] - base["wape"]).dropna()
        is_inner = g.index.get_level_values("protocol") == "inner_week"
        d_inner = delta[delta.index.get_level_values("protocol") == "inner_week"]
        d_folds = delta[delta.index.get_level_values("protocol") != "inner_week"]
        lines.append(
            f"| {variant} | {g[is_inner]['wape'].mean():.2%} ± {g[is_inner]['wape'].std():.2%} | "
            f"{g[is_inner]['wpe'].mean():+.2%} | {d_inner.mean():+.2%} ({int((d_inner < 0).sum())}/{len(d_inner)} better) | "
            f"{g[~is_inner]['wape'].mean():.2%} ± {g[~is_inner]['wape'].std():.2%} | "
            f"{d_folds.mean():+.2%} ({int((d_folds < 0).sum())}/{len(d_folds)} better) |"
        )
    table = "\n".join(lines)
    (out / "dlinear_ablation.md").write_text(
        "# DLinear ablation on validation data (inner week 06-12..06-18 and the 6 team folds)\n\n"
        "Raw-sales truth, stockout-free days, paper-style WAPE. Lower is better. Never uses calib or test.\n"
        "Selection uses the inner week (8 training origins per series at lookback 62). A fold trains on one\n"
        "window per series, all with the same dates, so calendar inputs (day of week, holidays) cannot be\n"
        "learned there: folds are a secondary check and the source of the contract's SD.\n\n"
        + table
        + "\n"
    )
    print(table)


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    main()
