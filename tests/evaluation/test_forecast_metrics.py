"""Forecast metrics on hand-computed frames, the scoring rules, and agreement with the paper's scoring."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from uncertainty_aware_planner.data.config import REPO_ROOT, load_yaml
from uncertainty_aware_planner.evaluation.forecast_metrics import (
    KEYS,
    METRIC_KEY,
    attach_slices,
    build_slices,
    check_metric_keys,
    interval_metrics,
    mase,
    metric_rows,
    paired_bootstrap_delta,
    pooled_wape,
    score_table,
    series_scale,
    top_misforecasts,
    training_window,
    wape_wpe,
)

O1, O2 = date(2024, 6, 25), date(2024, 6, 26)  # origins
T1, T2 = date(2024, 6, 27), date(2024, 6, 28)  # target days
META = {
    "model_id": "fc-tft-quantile",
    "imputation_run_id": "imp-dlinear-quantile-1024",
    "protocol": "project",
    "point": "estimated_demand",
    "fold": None,
    "seed": 2025,
    "split": "test",
}


def make_truth(levels: dict, overrides: dict | None = None) -> pd.DataFrame:
    """Every series-day of the data range at the series' level with no stockout, then the overrides
    {(store_id, product_id, day): (demand_total, stock_hour6_22_cnt)}."""
    overrides = overrides or {}
    rows = []
    for (store, product), level in levels.items():
        for day in pd.date_range("2024-03-28", "2024-07-02").date:
            demand, stock = overrides.get((store, product, day), (level, 0))
            rows.append(
                {
                    "store_id": store,
                    "product_id": product,
                    "date": day,
                    "demand_total": float(demand),
                    "stock_hour6_22_cnt": stock,
                }
            )
    return pd.DataFrame(rows)


def make_table(rows: list[tuple], **columns) -> pd.DataFrame:
    """Forecast table rows from (store_id, product_id, origin_date, horizon, estimated_demand) tuples."""
    table = pd.DataFrame(
        rows,
        columns=[
            "store_id",
            "product_id",
            "origin_date",
            "horizon",
            "estimated_demand",
        ],
    )
    table["target_date"] = [
        origin + timedelta(days=int(h))
        for origin, h in zip(table["origin_date"], table["horizon"])
    ]
    table["q025"] = np.nan
    table["q975"] = np.nan
    return table.assign(**columns)


def scored_frame(rows: list[tuple]) -> pd.DataFrame:
    """Scored rows from (store_id, product_id, origin_date, horizon, forecast, actual, slice) tuples."""
    frame = pd.DataFrame(
        rows,
        columns=[
            "store_id",
            "product_id",
            "origin_date",
            "horizon",
            "forecast",
            "actual",
            "slice",
        ],
    )
    frame["origin_date"] = pd.to_datetime(frame["origin_date"]).astype("datetime64[s]")
    frame["target_date"] = frame["origin_date"] + pd.to_timedelta(
        frame["horizon"], unit="D"
    )
    return frame


@pytest.fixture
def cells():
    """Two origins x two target days. (2, 3) is the only low_sales series and is out of stock on both target
    days, so low_sales is empty; cell (O1, T2) has no actual demand and is skipped."""
    truth = make_truth(
        {(1, 1): 2.0, (1, 2): 1.5, (2, 3): 0.5},
        {
            (1, 1, T1): (4, 0),
            (1, 2, T1): (2, 0),
            (2, 3, T1): (1, 3),
            (1, 1, T2): (0, 0),
            (1, 2, T2): (3, 0),
            (2, 3, T2): (0, 5),
        },
    )
    rows = [
        (1, 1, O1, 2, 3.0),  # T1: error -1
        (1, 1, O1, 3, 1.0),  # T2: error +1, in the skipped cell
        (2, 3, O1, 2, 1.0),  # stockout
        (2, 3, O1, 3, 1.0),  # stockout
        (1, 1, O2, 1, 5.0),  # T1: +1
        (1, 2, O2, 1, 1.0),  # T1: -1
        (1, 1, O2, 2, 1.0),  # T2: +1
        (1, 2, O2, 2, 2.0),  # T2: -1
        (2, 3, O2, 1, 1.0),  # stockout
        (2, 3, O2, 2, 1.0),  # stockout
    ]
    # actual_demand stands for an imputed history that differs from the recorded sales; it must be ignored.
    table = make_table(rows, actual_demand=100.0)
    return table, truth


def test_wape_wpe_by_cell(cells):
    table, truth = cells
    scored = attach_slices(score_table(table, truth), build_slices(truth))
    assert len(scored) == 6  # the four stockout rows are not scored
    # The truth, not the table's actual_demand.
    assert scored["actual"].tolist() == [4, 0, 4, 2, 0, 3]

    result = wape_wpe(scored).set_index(["slice", "metric"])
    # Cells: (O1,T1) 1/4; (O1,T2) skipped; (O2,T1) 2/6 and 0/6; (O2,T2) 2/3 and 0/3.
    assert result.loc[("all", "wape"), "value"] == pytest.approx(
        (1 / 4 + 2 / 6 + 2 / 3) / 3
    )
    assert result.loc[("all", "wpe"), "value"] == pytest.approx((-1 / 4 + 0 + 0) / 3)
    assert result.loc[("all", "wape"), "n"] == 5
    assert result.loc[("all", "wape"), "n_skipped"] == 1
    assert result.loc[("high_sales", "wape"), "value"] == pytest.approx(5 / 12)
    for metric in ("wape", "wpe"):
        assert np.isnan(result.loc[("low_sales", metric), "value"])
        assert result.loc[("low_sales", metric), "n"] == 0

    pooled = pooled_wape(scored).set_index("slice")
    assert pooled.loc["all", "value"] == pytest.approx(6 / 13)
    assert np.isnan(pooled.loc["low_sales", "value"])
    assert pooled.loc["low_sales", "n"] == 0


def test_score_table_checks_rows(cells):
    table, truth = cells
    with pytest.raises(ValueError, match="no truth row"):
        score_table(table.assign(store_id=table["store_id"].replace(2, 9)), truth)
    with pytest.raises(ValueError, match="duplicate"):
        score_table(pd.concat([table, table.iloc[:1]]), truth)
    with pytest.raises(ValueError, match="only one of q025 and q975"):
        score_table(table.assign(q025=[0.0] + [np.nan] * (len(table) - 1)), truth)
    with pytest.raises(ValueError, match="no estimated_demand forecast"):
        score_table(
            table.assign(estimated_demand=[np.nan] + [1.0] * (len(table) - 1)), truth
        )
    with pytest.raises(ValueError, match="origin_date \\+ horizon"):
        score_table(table.assign(horizon=table["horizon"] + 1), truth)
    # A censored run's actual_demand is the truth itself.
    with pytest.raises(ValueError, match="differs from the censored truth"):
        score_table(table, truth, check_actuals=True)
    actual = truth.set_index(KEYS + ["date"])["demand_total"]
    keys = list(zip(table["store_id"], table["product_id"], table["target_date"]))
    censored = table.assign(actual_demand=actual.loc[keys].to_numpy())
    assert len(score_table(censored, truth, check_actuals=True)) == 6


def test_build_slices_uses_the_psd_window():
    truth = make_truth(
        {(1, 1): 1.0, (1, 2): 0.9},
        {(1, 2, date(2024, 6, 26)): (500.0, 0)},  # test week: outside the psd window
    )
    slices = build_slices(truth).set_index(KEYS)
    assert slices.loc[(1, 1), "psd"] == pytest.approx(1.0)
    assert slices.loc[(1, 1), "slice"] == "high_sales"  # psd >= 1
    assert slices.loc[(1, 2), "psd"] == pytest.approx(0.9)
    assert slices.loc[(1, 2), "slice"] == "low_sales"


def test_mase_drops_zero_scale_series_and_scales_on_the_training_window():
    window = {date(2024, 6, d): v for d, v in zip(range(1, 6), [1, 3, 2, 2, 4])}
    overrides = {(1, 1, day): (value, 0) for day, value in window.items()}
    overrides[(1, 1, date(2024, 5, 31))] = (100, 0)  # outside the window
    overrides[(1, 1, date(2024, 6, 6))] = (50, 0)  # outside the window
    overrides.update(
        {(2, 3, date(2024, 6, d)): (v, 3) for d, v in zip(range(1, 6), [0, 1, 0, 1, 0])}
    )
    truth = make_truth({(1, 1): 7.0, (1, 2): 5.0, (2, 3): 0.0}, overrides)

    scale = series_scale(truth, date(2024, 6, 1), date(2024, 6, 5))
    assert scale.loc[(1, 1)] == pytest.approx(5 / 4)  # |2|, |-1|, |0|, |2|
    assert scale.loc[(1, 2)] == 0  # constant: dropped
    # Stockout days still count for the scale.
    assert scale.loc[(2, 3)] == pytest.approx(1.0)

    scored = scored_frame(
        [
            (1, 1, O1, 1, 5.0, 4.0, "high_sales"),
            (1, 1, O1, 2, 1.0, 4.0, "high_sales"),
            (1, 2, O1, 1, 2.0, 1.0, "high_sales"),
            (2, 3, O1, 1, 0.5, 0.0, "low_sales"),
            (2, 3, O1, 2, 0.5, 1.0, "low_sales"),
        ]
    )
    result = mase(scored, scale).set_index("slice")
    # (1, 1): mean |e| 2 / 1.25 = 1.6; (2, 3): 0.5 / 1 = 0.5; (1, 2) dropped.
    assert result.loc["all", "value"] == pytest.approx((1.6 + 0.5) / 2)
    assert result.loc["all", ["n", "n_skipped"]].tolist() == [2, 1]
    assert result.loc["high_sales", "value"] == pytest.approx(1.6)
    assert result.loc["high_sales", ["n", "n_skipped"]].tolist() == [1, 1]
    assert result.loc["low_sales", "value"] == pytest.approx(0.5)
    assert result.loc["low_sales", ["n", "n_skipped"]].tolist() == [1, 0]


def test_training_windows():
    splits = load_yaml("configs/splits.yaml")
    fold1 = training_window(splits, 1, "project")
    assert (fold1.start, fold1.end) == (date(2024, 4, 1), date(2024, 4, 28))
    final = training_window(splits, None, "project")
    assert (final.start, final.end) == (date(2024, 3, 28), date(2024, 6, 18))
    paper = training_window(splits, None, "paper")
    assert (paper.start, paper.end) == (date(2024, 3, 28), date(2024, 6, 25))


def test_interval_metrics_inclusive_bounds():
    scored = scored_frame(
        [
            (1, 1, O1, 1, 0.0, 0.0, "low_sales"),  # [0, 0] holds A = 0
            (1, 1, O1, 2, 2.5, 2.0, "low_sales"),  # on the lower bound
            (1, 1, O1, 3, 2.5, 3.0, "low_sales"),  # on the upper bound
            (1, 1, O1, 4, 2.5, 4.0, "low_sales"),  # outside
            (1, 1, O1, 5, 2.5, 1.0, "low_sales"),  # no interval
        ]
    ).assign(
        q025=[0.0, 2.0, 2.0, 2.0, np.nan],
        q975=[0.0, 3.0, 3.0, 3.0, np.nan],
        q025_cal=[0.0, np.nan, np.nan, np.nan, np.nan],
        q975_cal=[1.0, np.nan, np.nan, np.nan, np.nan],
    )
    raw = interval_metrics(scored).set_index(["slice", "metric"])
    assert raw.loc[("all", "picp95"), "value"] == pytest.approx(3 / 4)
    assert raw.loc[("all", "mpiw95"), "value"] == pytest.approx(3 / 4)
    assert raw.loc[("all", "picp95"), ["n", "n_skipped"]].tolist() == [4, 1]
    assert np.isnan(raw.loc[("high_sales", "picp95"), "value"])
    calibrated = interval_metrics(scored, calibrated=True).set_index(
        ["slice", "metric"]
    )
    assert calibrated.loc[("all", "picp95"), "value"] == 1.0
    assert calibrated.loc[("all", "mpiw95"), "n"] == 1

    scale = pd.Series([1.0], index=pd.MultiIndex.from_tuples([(1, 1)], names=KEYS))
    rows = metric_rows(scored, scale, META)
    assert set(rows.loc[rows["calibrated"], "metric"]) == {"picp95", "mpiw95"}
    check_metric_keys(rows)

    point_only = scored.assign(
        q025=np.nan, q975=np.nan, q025_cal=np.nan, q975_cal=np.nan
    )
    rows = metric_rows(point_only, scale, META)
    assert set(rows["metric"]) == {"wape", "wpe", "mase"}
    assert len(rows) == 9 and not rows["calibrated"].any()


def test_metric_rows_follow_the_contract(cells):
    table, truth = cells
    scored = attach_slices(score_table(table, truth), build_slices(truth))
    scale = series_scale(truth, date(2024, 3, 28), date(2024, 6, 18))
    rows = metric_rows(scored, scale, META)
    assert set(rows["slice"]) == {"all", "low_sales", "high_sales"}
    assert rows["fold"].dtype == "Int64" and rows["fold"].isna().all()
    assert rows["calibrated"].dtype == bool and rows["n"].dtype == "int64"
    assert not rows.duplicated(METRIC_KEY).any()
    check_metric_keys(rows)
    with pytest.raises(ValueError, match="share a key"):
        check_metric_keys(pd.concat([rows, rows.iloc[:1]]))
    # Another point column for the same run is a different key.
    check_metric_keys(pd.concat([rows, rows.assign(point="q500")]))


def test_wape_matches_paper_scoring(monkeypatch):
    """One origin and 7 target days: the paper's mean of daily WAPEs, with its psd groups."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "experiments" / "vishnu"))
    from paper_wape import paper_metrics

    rng = np.random.default_rng(0)
    series = [(store, product) for store in range(6) for product in range(10)]
    levels = {pair: (0.5 if i % 2 else 3.0) for i, pair in enumerate(series)}
    truth = make_truth(levels)
    level = [levels[pair] for pair in zip(truth["store_id"], truth["product_id"])]
    truth["demand_total"] = rng.poisson(level).astype(float)
    test_week = pd.to_datetime(truth["date"]) >= pd.Timestamp("2024-06-26")
    truth.loc[test_week, "stock_hour6_22_cnt"] = np.where(
        rng.random(test_week.sum()) < 0.3, rng.integers(1, 5, test_week.sum()), 0
    )

    quantiles = pd.DataFrame(
        [(s, p, h) for s, p in series for h in range(1, 8)],
        columns=["store_id", "product_id", "horizon"],
    )
    quantiles["origin_date"] = O1
    quantiles["target_date"] = [O1 + timedelta(days=h) for h in quantiles["horizon"]]
    quantiles["q500"] = rng.gamma(2.0, 1.0, len(quantiles))

    expected = paper_metrics(quantiles, truth).set_index("group")
    scored = attach_slices(
        score_table(quantiles, truth, point="q500"), build_slices(truth)
    )
    result = wape_wpe(scored).set_index(["slice", "metric"])
    for slice_, group in (
        ("all", "psd>=0"),
        ("high_sales", "psd>=1"),
        ("low_sales", "psd<1"),
    ):
        assert result.loc[(slice_, "wape"), "n_skipped"] == 0
        assert result.loc[(slice_, "wape"), "value"] == pytest.approx(
            expected.loc[group, "wape"], rel=1e-12
        )
        assert result.loc[(slice_, "wpe"), "value"] == pytest.approx(
            expected.loc[group, "wpe"], rel=1e-12
        )
        assert result.loc[(slice_, "wape"), "n"] == expected.loc[group, "rows"]


def test_paired_bootstrap_delta():
    rows = [
        (store, product, O1, h, 0.0, float(1 + (store + product + h) % 4), "high_sales")
        for store in range(5)
        for product in range(8)
        for h in range(1, 8)
    ]
    truth_like = scored_frame(rows)
    perfect = truth_like.assign(forecast=truth_like["actual"])
    biased = truth_like.assign(forecast=truth_like["actual"] + 1)

    result = paired_bootstrap_delta(perfect, biased, n_boot=200, seed=1)
    assert result["wape_a"] == 0
    assert result["delta"] == pytest.approx(-result["wape_b"])
    assert result["ci_low"] <= result["delta"] <= result["ci_high"] < 0
    assert (result["n_series"], result["n_rows"]) == (40, 280)

    same = paired_bootstrap_delta(biased, biased, n_boot=50)
    assert (same["delta"], same["ci_low"], same["ci_high"]) == (0, 0, 0)
    # Two seeds that average to the truth score like the perfect forecast.
    seeds = pd.concat(
        [truth_like.assign(forecast=truth_like["actual"] + d) for d in (-0.5, 0.5)]
    )
    averaged = paired_bootstrap_delta(seeds, perfect, n_boot=50)
    assert averaged["wape_a"] == pytest.approx(0)
    assert averaged["delta"] == pytest.approx(0)


def test_top_misforecasts_average_seeds():
    rows = [
        (1, 1, O1, 1, 0.0, 1.0, "high_sales"),
        (1, 1, O1, 2, 0.0, 5.0, "high_sales"),
        (2, 3, O1, 1, 0.0, 2.0, "low_sales"),
        (2, 3, O1, 2, 0.0, 0.0, "low_sales"),
    ]
    first = scored_frame(rows).assign(forecast=[1.0, 1.0, 6.0, 0.0])
    second = scored_frame(rows).assign(forecast=[1.0, 3.0, 4.0, 2.0])
    top = top_misforecasts(pd.concat([first, second]), k=2)
    best = top[top["group"] == "all"]
    # Seed means: (1,1,h2) 2 vs 5 -> 3; (2,3,h1) 5 vs 2 -> 3; (2,3,h2) 1 vs 0 -> 1.
    assert best["abs_error"].tolist() == [3.0, 3.0]
    assert best["rank"].tolist() == [1, 2]
    low = top[top["group"] == "low_sales"]
    assert low["product_id"].tolist() == [3, 3]
    assert low["abs_error"].tolist() == [3.0, 1.0]
    assert (top["n_seeds"] == 2).all()
    assert isinstance(top["target_date"].iloc[0], date)
