"""The forecast evaluation command runs end to end on synthetic tables and reproduces the real TFT scores."""

import glob
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uncertainty_aware_planner.data.config import REPO_ROOT
from uncertainty_aware_planner.evaluation.forecast_metrics import (
    METRIC_KEY,
    read_metrics,
)
from uncertainty_aware_planner.models.forecasting.forecast_table import COLUMNS
from uncertainty_aware_planner.pipeline.evaluate_forecasting import evaluate, main

CENSORED = "imp-none-censored"
# (store_id, product_id): mean daily demand; two high_sales and two low_sales series.
SERIES = {(0, 1): 3.0, (0, 2): 0.5, (1, 1): 2.0, (1, 3): 0.4}
FOLD1_ORIGINS = [date(2024, 4, 28), date(2024, 5, 5)]  # fold 1 val window: 04-29..05-26


def write_truth(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for (store, product), level in SERIES.items():
        for day in pd.date_range("2024-03-28", "2024-07-02").date:
            rows.append(
                {
                    "store_id": store,
                    "product_id": product,
                    "date": day,
                    "imputation_run_id": CENSORED,
                    "demand_total": float(rng.poisson(level)),
                    "stock_hour6_22_cnt": int(rng.integers(1, 4))
                    if rng.random() < 0.2
                    else 0,
                }
            )
    truth = pd.DataFrame(rows)
    truth.to_parquet(path, index=False)
    return truth


def write_run(root, truth, model_id, seed, fold, split, origins, point_only=False):
    """One forecast table and its run-directory quantile file, laid out as pipeline/forecast.py does."""
    rng = np.random.default_rng([seed, fold or 0, len(model_id)])
    actual = truth.set_index(["store_id", "product_id", "date"])["demand_total"]
    rows = []
    for store, product in SERIES:
        for origin in origins:
            for horizon in range(1, 8):
                target = origin + timedelta(days=horizon)
                q500 = max(actual[(store, product, target)] + rng.normal(0, 1), 0)
                rows.append((store, product, horizon, origin, target, q500))
    quantiles = pd.DataFrame(
        rows,
        columns=[
            "store_id",
            "product_id",
            "horizon",
            "origin_date",
            "target_date",
            "q500",
        ],
    )
    quantiles["horizon"] = quantiles["horizon"].astype("uint8")
    quantiles["q500"] = quantiles["q500"].astype("float32")
    if not point_only:
        quantiles["q025"] = (quantiles["q500"] * 0.5).astype("float32")
        quantiles["q975"] = (quantiles["q500"] * 1.5 + 1).astype("float32")

    table = quantiles.copy()
    for column in COLUMNS:
        if column not in table:
            table[column] = 0
    table["date"] = table["target_date"]
    table["imputation_run_id"] = CENSORED
    table["forecast_model_id"] = model_id
    table["forecast_seed"] = pd.array([seed] * len(table), dtype="UInt32")
    table["forecast_fold"] = pd.array([fold] * len(table), dtype="UInt8")
    table["forecast_split"] = split
    table["forecast_run_id"] = f"{model_id}-{seed}"
    if point_only:
        table["q025"] = table["q975"] = np.nan
        table["estimated_demand"] = table["q500"].astype("float64")
    else:
        table["q025"] = table["q025"].astype("float64")
        table["q975"] = table["q975"].astype("float64")
        table["estimated_demand"] = (table["q025"] + table["q975"]) / 2
    keys = list(zip(table["store_id"], table["product_id"], table["target_date"]))
    table["actual_demand"] = actual.loc[keys].to_numpy()
    for column in ("q025_cal", "q975_cal", "qhat95"):
        table[column] = np.nan

    rid = f"{model_id}-{seed}"
    name = f"fold{fold}" if fold is not None else f"final_{split}"
    table_dir = root / "forecast" / CENSORED / rid
    run_dir = (
        root
        / "runs"
        / CENSORED
        / rid
        / (f"fold{fold}" if fold is not None else "final")
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    table[COLUMNS].to_parquet(table_dir / f"{name}.parquet", index=False)
    quantiles.to_parquet(run_dir / f"quantiles_{split}.parquet", index=False)


@pytest.fixture
def world(tmp_path):
    truth = write_truth(tmp_path / "truth.parquet")
    for seed in (2025, 2026):
        write_run(tmp_path, truth, "fc-tft-quantile", seed, 1, "val", FOLD1_ORIGINS)
        write_run(
            tmp_path, truth, "fc-tft-quantile", seed, None, "calib", [date(2024, 6, 18)]
        )
        write_run(
            tmp_path, truth, "fc-tft-quantile", seed, None, "test", [date(2024, 6, 25)]
        )
        write_run(
            tmp_path,
            truth,
            "fc-dlinear-baseline",
            seed,
            None,
            "test",
            [date(2024, 6, 25)],
            point_only=True,
        )
    return tmp_path


def run_cli(root: Path, *extra: str) -> None:
    args = ["--tables", str(root / "forecast" / "**" / "*.parquet")]
    args += [
        "--truth",
        str(root / "truth.parquet"),
        "--slices",
        str(root / "slices.parquet"),
    ]
    args += ["--quantiles-root", str(root / "runs"), "--out", str(root / "results")]
    assert main(args + list(extra)) == 0


def test_cli_end_to_end(world, capsys):
    run_cli(world, "--baseline-model", "fc-dlinear-baseline")
    out = world / "results" / CENSORED / "project"
    for name in (
        "metrics.csv",
        "pooled_wape.csv",
        "summary.csv",
        "summary.md",
        "top_misforecasts.csv",
    ):
        assert (out / name).exists()
    assert "summary.md" in capsys.readouterr().out
    manifest = json.loads((world / "slices.parquet.manifest.json").read_text())
    assert (manifest["start"], manifest["end"], manifest["rows"]) == (
        "2024-03-28",
        "2024-06-25",
        4,
    )

    metrics = read_metrics(out / "metrics.csv")
    assert not metrics.duplicated(METRIC_KEY).any()
    # TFT: 6 tables x 15 rows (wape, wpe, mase, picp95, mpiw95 per slice); point-only DLinear: 2 x 9.
    assert len(metrics) == 6 * 15 + 2 * 9
    dlinear = metrics[metrics["model_id"] == "fc-dlinear-baseline"]
    assert set(dlinear["metric"]) == {"wape", "wpe", "mase"}
    assert set(metrics["split"]) == {"val", "calib", "test"}
    assert metrics.loc[metrics["split"] == "val", "fold"].eq(1).all()

    summary = pd.read_csv(out / "summary.csv").set_index(["model_id", "slice"])
    tft = summary.loc[("fc-tft-quantile", "all")]
    assert (tft["n_seeds"], tft["cv_n"]) == (2, 2)
    assert tft["published_wape"] == pytest.approx(0.3175)
    assert summary.loc[
        ("fc-dlinear-baseline", "all"), "published_wape"
    ] == pytest.approx(0.3156)
    assert "Paired bootstrap" in (out / "summary.md").read_text()
    comparisons = pd.read_csv(out / "comparisons.csv")
    assert comparisons[["model_id", "baseline_model_id"]].values.tolist() == [
        ["fc-tft-quantile", "fc-dlinear-baseline"]
    ]
    top = pd.read_csv(out / "top_misforecasts.csv")
    assert set(top["group"]) == {"all", "low_sales"} and (top["n_seeds"] == 2).all()
    ledger = pd.read_csv(world / "results" / "test_ledger.csv")
    assert sorted(ledger["model_id"]) == ["fc-dlinear-baseline", "fc-tft-quantile"]
    assert (ledger["seeds"] == "2025 2026").all() and (ledger["n_tables"] == 2).all()

    # q500 from the run directories: the rows accumulate next to the estimated_demand ones.
    run_cli(world, "--point", "q500")
    metrics = read_metrics(out / "metrics.csv")
    assert len(metrics) == 2 * (6 * 15 + 2 * 9)
    test_wape = metrics[(metrics["split"] == "test") & (metrics["metric"] == "wape")]
    by_point = test_wape.pivot_table(
        index=["model_id", "seed", "slice"], columns="point", values="value"
    )
    # A point model's estimated_demand is its q500; a quantile model's is the interval midpoint.
    dlinear_wape = by_point.loc["fc-dlinear-baseline"]
    np.testing.assert_allclose(dlinear_wape["q500"], dlinear_wape["estimated_demand"])
    assert not np.allclose(
        by_point.loc["fc-tft-quantile", "q500"],
        by_point.loc["fc-tft-quantile", "estimated_demand"],
    )
    assert len(pd.read_csv(world / "results" / "test_ledger.csv")) == 4

    # Re-scoring replaces rows instead of adding them; --no-ledger leaves the ledger alone.
    run_cli(world, "--point", "q500", "--no-ledger")
    again = read_metrics(out / "metrics.csv")
    assert len(again) == len(metrics) and not again.duplicated(METRIC_KEY).any()
    assert len(pd.read_csv(world / "results" / "test_ledger.csv")) == 4


def test_paper_protocol_borrows_project_folds(world):
    run_cli(world, "--no-ledger")
    tables = sorted(
        (world / "forecast").glob("**/fc-tft-quantile-*/final_test.parquet")
    )
    result = evaluate(
        tables,
        truth_path=world / "truth.parquet",
        slices_path=world / "slices.parquet",
        protocol="paper",
        out_dir=world / "results",
        ledger=False,
    )
    summary = result["summary"].set_index("slice")
    assert summary.loc["all", "protocol"] == "paper"
    assert summary.loc["all", "cv_n"] == 2  # the project protocol's fold rows
    paper = read_metrics(world / "results" / CENSORED / "paper" / "metrics.csv")
    project = read_metrics(world / "results" / CENSORED / "project" / "metrics.csv")
    # The paper protocol scales MASE on 03-28..06-25 instead of 03-28..06-18; WAPE is unchanged.
    key = ["model_id", "seed", "split", "slice", "metric"]
    both = paper.merge(project[project["point"] == "estimated_demand"], on=key)
    wape = both[both["metric"] == "wape"]
    np.testing.assert_allclose(wape["value_x"], wape["value_y"])
    mase = both[both["metric"] == "mase"]
    assert not np.allclose(mase["value_x"], mase["value_y"])


def test_cli_errors(world):
    with pytest.raises(FileNotFoundError, match="no parquet files"):
        main(["--tables", str(world / "nothing" / "**" / "*.parquet")])
    for path in (world / "runs").glob(
        "**/fc-tft-quantile-2025/final/quantiles_test.parquet"
    ):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="quantiles_test.parquet"):
        run_cli(world, "--point", "q500", "--no-ledger")
    with pytest.raises(ValueError, match="baseline model"):
        run_cli(world, "--baseline-model", "fc-unknown", "--no-ledger")


# Real-data regression: the TFT runs scored by experiments/vishnu/paper_wape.py (q500) and the contract
# point forecast (estimated_demand, the 95% interval midpoint). Skipped when the files are absent.
TRUTH = REPO_ROOT / "data" / "processed" / "demand_daily" / f"{CENSORED}.parquet"
SEEDS = (2025, 2026, 2027)


def real_table(pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(REPO_ROOT / pattern), recursive=True))
    return Path(matches[0]) if matches else None


PROJECT_TABLES = [
    real_table(f"data/processed/forecast/**/fc-tft-quantile-{seed}/final_test.parquet")
    for seed in SEEDS
]
PAPER_TABLE = real_table(
    "data/processed/forecast_paper_split/**/fc-tft-quantile-2025/final_test.parquet"
)
QUANTILES = REPO_ROOT / "data" / "runs" / "forecasting"
PAPER_QUANTILES = REPO_ROOT / "data" / "runs" / "forecasting_paper_split"


def scored_wape(tables, tmp_path, **options) -> pd.Series:
    result = evaluate(
        tables,
        truth_path=TRUTH,
        slices_path=tmp_path / "slices.parquet",
        out_dir=tmp_path / "results",
        ledger=False,
        **options,
    )
    metrics = read_metrics(result["metrics"][CENSORED])
    wape = metrics[
        (metrics["split"] == "test")
        & (metrics["slice"] == "all")
        & (metrics["metric"] == "wape")
    ]
    return wape.set_index("seed")["value"]


@pytest.mark.skipif(
    not TRUTH.exists() or None in PROJECT_TABLES or not QUANTILES.exists(),
    reason="real TFT test tables not available",
)
@pytest.mark.parametrize(
    "point, expected",
    [
        ("q500", (0.3312, 0.3357, 0.3362)),
        ("estimated_demand", (0.3409, 0.3351, 0.3503)),
    ],
)
def test_real_tft_test_week(tmp_path, point, expected):
    wape = scored_wape(PROJECT_TABLES, tmp_path, point=point, quantiles_root=QUANTILES)
    for seed, value in zip(SEEDS, expected):
        assert wape[seed] == pytest.approx(value, abs=1e-4)


@pytest.mark.skipif(
    not TRUTH.exists() or PAPER_TABLE is None or not PAPER_QUANTILES.exists(),
    reason="real paper-split TFT table not available",
)
def test_real_tft_paper_split(tmp_path):
    wape = scored_wape(
        [PAPER_TABLE],
        tmp_path,
        point="q500",
        quantiles_root=PAPER_QUANTILES,
        protocol="paper",
    )
    assert wape[2025] == pytest.approx(0.3173, abs=1e-4)
