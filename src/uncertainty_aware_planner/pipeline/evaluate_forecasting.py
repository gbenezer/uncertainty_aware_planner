"""Score forecast tables (CONTRACTS.md 2.5) against raw recorded sales and write 2.10 metrics and a summary.

    python -m uncertainty_aware_planner.pipeline.evaluate_forecasting \
        --tables "data/processed/forecast/imp-none-censored/**/*.parquet" \
        [--point q500] [--protocol project] [--baseline-model fc-dlinear-baseline] [--no-ledger]

Every table is scored against the censored demand_daily's demand_total on target days without a stockout
between 06:00 and 22:00, whatever demand history the model was trained on. --point q500 takes the median
from the run directory: <quantiles-root>/**/<run_id>/{fold<k>,final}/quantiles_<split>.parquet.

Outputs go to <out>/<imputation_run_id>/<protocol>/: metrics.csv (2.10 rows; a re-evaluated table replaces
its rows), pooled_wape.csv, summary.csv, summary.md, top_misforecasts.csv and, with --baseline-model,
comparisons.csv. Scoring test-week tables appends to <out>/test_ledger.csv unless --no-ledger. Evaluations
writing into the same --out at the same time take turns through a lock on <out>/.evaluate.lock. The sales
slices are built once into --slices (with a manifest) when that file is missing.
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # not POSIX (Windows): concurrent evaluations are not serialised
    fcntl = None

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from uncertainty_aware_planner.data.config import load_config, resolve
from uncertainty_aware_planner.data.outputs import file_sha256
from uncertainty_aware_planner.evaluation.forecast_metrics import (
    CENSORED_RUN,
    FEATURE,
    POINTS,
    PROTOCOLS,
    ROW_KEY,
    SLICE_END,
    SLICE_START,
    SPLITS,
    as_day,
    attach_slices,
    build_slices,
    check_metric_keys,
    has_bounds,
    load_truth,
    metric_rows,
    paired_bootstrap_delta,
    pooled_wape,
    read_metrics,
    read_parquet,
    score_table,
    series_scale,
    top_misforecasts,
    training_window,
)
from uncertainty_aware_planner.evaluation.report import (
    DEFAULT_TARGETS,
    GROUP,
    append_ledger,
    combined_sha256,
    load_targets,
    short_table,
    summarize,
    write_summary,
)
from uncertainty_aware_planner.models.forecasting.forecast_table import run_id

DEFAULT_TRUTH = "data/processed/demand_daily/imp-none-censored.parquet"
DEFAULT_SLICES = "data/processed/slices.parquet"
DEFAULT_QUANTILES = "data/runs/forecasting"
DEFAULT_OUT = "results/forecasting"
DEFAULT_DATA_CONFIG = "configs/data.yaml"
# Forecast table column -> run field; each is constant within one table.
IDENTITY = {
    "forecast_model_id": "model_id",
    "forecast_seed": "seed",
    "forecast_fold": "fold",
    "forecast_split": "split",
    "imputation_run_id": "imputation_run_id",
}
TABLE_COLUMNS = ROW_KEY + [
    "target_date",
    "estimated_demand",
    "q025",
    "q975",
    "q025_cal",
    "q975_cal",
]
RUN_KEY = [
    "feature",
    "model_id",
    "imputation_run_id",
    "protocol",
    "point",
    "fold",
    "seed",
    "split",
]
POOLED_COLUMNS = RUN_KEY + ["slice", "value", "n"]
TEST_COLUMNS = ROW_KEY + ["target_date", "forecast", "actual", "slice"]
BASELINE = ["baseline_model_id", "baseline_imputation_run_id"]
FOLDER_OUTPUTS = (
    "out_dirs",
    "metrics",
    "summary_csv",
    "summary_md",
    "top_misforecasts",
)
QUANTILE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class TableInfo:
    """A forecast table file and the run part it holds."""

    path: Path
    model_id: str
    seed: int
    fold: int | None
    split: str
    imputation_run_id: str


def expand_tables(patterns: list[str]) -> list[Path]:
    """Parquet files matching the glob patterns (recursive **, relative to the repo root), without repeats;
    raises when a pattern matches none."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(resolve(pattern)), recursive=True))
        files = [
            Path(m) for m in matches if m.endswith(".parquet") and Path(m).is_file()
        ]
        if not files:
            raise FileNotFoundError(f"no parquet files match {pattern}")
        paths += files
    return list(dict.fromkeys(paths))


def table_info(path: Path) -> TableInfo:
    """Run identity from a forecast table's columns; raises unless each is constant over the table."""
    names = pq.read_schema(path).names
    missing = [column for column in IDENTITY if column not in names]
    if missing:
        raise ValueError(f"{path} is not a forecast table (no {missing} columns)")
    strings = ["forecast_model_id", "forecast_split", "imputation_run_id"]
    table = pq.read_table(path, columns=list(IDENTITY), read_dictionary=strings)
    values = {}
    for column, field in IDENTITY.items():
        unique = pc.unique(table.column(column)).to_pylist()
        if len(unique) != 1:
            raise ValueError(
                f"{path}: {column} takes {len(unique)} values, expected one"
            )
        if unique[0] is None and field != "fold":
            raise ValueError(f"{path}: {column} is null")
        values[field] = unique[0]
    fold, split = values["fold"], values["split"]
    if split not in SPLITS:
        raise ValueError(f"{path}: split {split!r} is not one of {SPLITS}")
    if (fold is None) != (split != "val"):
        raise ValueError(
            f"{path}: fold tables have split val, final tables calib or test"
        )
    return TableInfo(
        path=path,
        model_id=str(values["model_id"]),
        seed=int(values["seed"]),
        fold=None if fold is None else int(fold),
        split=split,
        imputation_run_id=str(values["imputation_run_id"]),
    )


def find_quantile_file(root: Path, info: TableInfo) -> Path:
    """The run directory's quantile file for a table: <root>/**/<run_id>/fold<k>/quantiles_val.parquet or
    <root>/**/<run_id>/final/quantiles_<split>.parquet. Among several, the one under the table's
    imputation_run_id."""
    part = f"fold{info.fold}" if info.fold is not None else "final"
    rid = run_id(info.model_id, info.seed)
    pattern = Path(glob.escape(str(root))) / "**" / glob.escape(rid) / part
    matches = sorted(
        Path(m)
        for m in glob.glob(
            str(pattern / f"quantiles_{info.split}.parquet"), recursive=True
        )
    )
    if not matches:
        raise FileNotFoundError(
            f"no quantiles_{info.split}.parquet for {rid}/{part} under {root}"
        )
    if len(matches) > 1:
        matches = [m for m in matches if info.imputation_run_id in m.parts]
        if len(matches) != 1:
            raise ValueError(
                f"several quantile files for {rid}/{part} under {root}, none or several of them "
                f"under {info.imputation_run_id}"
            )
    return matches[0]


def join_q500(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add q500 from a run directory's quantile file, after checking that the file belongs to the table: its
    q025 and q975 (for point models, q500 against estimated_demand) must equal the table's."""
    names = pq.read_schema(path).names
    if "q500" not in names:
        raise ValueError(f"{path} has no q500 column")
    checked = ["q025", "q975"] if has_bounds(table) else []
    if not set(checked) <= set(names):
        raise ValueError(f"{path} has no q025/q975 to check against the table")
    quantiles = read_parquet(path, ROW_KEY + ["q500"] + checked)
    quantiles["origin_date"] = as_day(quantiles["origin_date"])
    joined = table.assign(origin_date=as_day(table["origin_date"])).merge(
        quantiles,
        on=ROW_KEY,
        how="left",
        suffixes=("", "_file"),
        validate="many_to_one",
    )
    pairs = [(c, f"{c}_file") for c in checked] or [("estimated_demand", "q500")]
    for mine, theirs in pairs:
        both = joined[mine].notna() & joined[theirs].notna()
        if not np.allclose(
            joined.loc[both, mine].to_numpy(dtype="float64"),
            joined.loc[both, theirs].to_numpy(dtype="float64"),
            rtol=QUANTILE_TOLERANCE,
            atol=QUANTILE_TOLERANCE,
        ):
            raise ValueError(
                f"{path} does not match the table: {theirs} differs from {mine}"
            )
    return joined.drop(columns=[f"{c}_file" for c in checked])


def read_table(info: TableInfo, point: str, quantiles_root: str | Path) -> pd.DataFrame:
    """The columns of a forecast table that scoring needs (plus q500 from the run directory for point q500)."""
    names = pq.read_schema(info.path).names
    columns = [column for column in TABLE_COLUMNS if column in names]
    if info.imputation_run_id == CENSORED_RUN:
        columns.append("actual_demand")
    table = read_parquet(info.path, columns)
    if point == "q500":
        table = join_q500(table, find_quantile_file(resolve(quantiles_root), info))
    return table


def load_slices(path: Path, truth: pd.DataFrame, truth_path: Path) -> pd.DataFrame:
    """slices.parquet, built from the truth with <path>.manifest.json when missing. Warns when an existing
    file was built from another truth file or psd window."""
    manifest_path = path.with_name(path.name + ".manifest.json")
    truth_sha = file_sha256(truth_path)
    if path.exists():
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            built = (
                manifest.get("truth_sha256"),
                manifest.get("start"),
                manifest.get("end"),
            )
            if built != (truth_sha, str(SLICE_START), str(SLICE_END)):
                warnings.warn(
                    f"{path} was built from another truth file or window; delete it to rebuild",
                    stacklevel=2,
                )
        return pd.read_parquet(path)

    slices = build_slices(truth, SLICE_START, SLICE_END)
    path.parent.mkdir(parents=True, exist_ok=True)
    slices.to_parquet(path, index=False)
    manifest = {
        "output": str(path),
        "sha256": file_sha256(path),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "truth": str(truth_path),
        "truth_sha256": truth_sha,
        "start": str(SLICE_START),
        "end": str(SLICE_END),
        "rows": len(slices),
        "low_sales": int((slices["slice"] == "low_sales").sum()),
        "high_sales": int((slices["slice"] == "high_sales").sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return slices


@contextmanager
def _output_lock(out_root: Path):
    """Let evaluations that share an output folder (e.g. two experiment jobs on one imputation run) take
    turns at the read-merge-write of metrics.csv and the ledger: an advisory lock on <out>/.evaluate.lock."""
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / ".evaluate.lock", "a") as handle:
        if fcntl is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX)
            except OSError:
                warnings.warn(
                    f"no file locks under {out_root}: do not evaluate into it concurrently",
                    stacklevel=3,
                )
        yield  # closing the file releases the lock


def _pooled_frame(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows[POOLED_COLUMNS].copy()
    out["fold"] = out["fold"].astype("Int64")
    out["seed"] = out["seed"].astype("int64")
    out["value"] = out["value"].astype("float64")
    out["n"] = out["n"].astype("int64")
    return out.reset_index(drop=True)


def _read_pooled(path: Path) -> pd.DataFrame:
    return _pooled_frame(pd.read_csv(path, dtype={"fold": "Int64"}))


def _merge_rows(path: Path, new: pd.DataFrame, key: list[str], read) -> pd.DataFrame:
    """`new` plus the rows of the file at `path` whose key does not appear in `new`."""
    if not path.exists():
        return new.reset_index(drop=True)
    old = read(path)
    replaced = old.merge(new[key].drop_duplicates(), on=key, how="left", indicator=True)
    keep = replaced["_merge"].eq("left_only").to_numpy()
    merged = pd.concat([old[keep], new], ignore_index=True)
    order = [
        column
        for column in ("model_id", "point", "split", "fold", "seed")
        if column in key
    ]
    return merged.sort_values(order, kind="stable", na_position="last").reset_index(
        drop=True
    )


def _comparisons(tests: dict, baseline_model: str) -> pd.DataFrame:
    """Paired bootstrap of every scored test group against each group of the baseline model."""
    rows = []
    for base, base_parts in tests.items():
        if base[0] != baseline_model:
            continue
        base_rows = pd.concat([frame for _, frame in base_parts], ignore_index=True)
        for group, parts in tests.items():
            if group == base:
                continue
            delta = paired_bootstrap_delta(
                pd.concat([frame for _, frame in parts], ignore_index=True), base_rows
            )
            rows.append(
                {**dict(zip(GROUP, group)), **dict(zip(BASELINE, base[:2])), **delta}
            )
    columns = GROUP + BASELINE + ["wape_a", "wape_b", "delta", "ci_low", "ci_high"]
    return pd.DataFrame(rows, columns=columns + ["n_series", "n_rows", "n_boot"])


def _merge_comparisons(path: Path, new: pd.DataFrame, rescored: set) -> pd.DataFrame:
    """Earlier comparisons plus `new`, dropping earlier ones whose model or baseline was re-scored now."""
    if not path.exists():
        return new
    old = pd.read_csv(path)
    model = zip(*(old[column] for column in GROUP))
    base = zip(old[BASELINE[0]], old[BASELINE[1]], old["protocol"], old["point"])
    keep = np.array(
        [m not in rescored and b not in rescored for m, b in zip(model, base)], bool
    )
    return pd.concat([old[keep], new], ignore_index=True)


def _write_folder(
    folder: Path,
    metrics: pd.DataFrame,
    pooled: pd.DataFrame,
    comparisons: pd.DataFrame | None,
    tests: dict,
    targets: dict,
) -> tuple[dict, pd.DataFrame]:
    """Merge this call's rows into one <out>/<imputation_run_id>/<protocol>/ folder and rewrite its summary;
    returns the written paths (FOLDER_OUTPUTS keys) and the summary."""
    imputation_run_id, protocol = folder.parent.name, folder.name
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"out_dirs": folder, "metrics": folder / "metrics.csv"}
    all_metrics = _merge_rows(paths["metrics"], metrics, RUN_KEY, read_metrics)
    check_metric_keys(all_metrics)
    all_metrics.to_csv(paths["metrics"], index=False)
    pooled_path = folder / "pooled_wape.csv"
    all_pooled = _merge_rows(pooled_path, pooled, RUN_KEY, _read_pooled)
    all_pooled.to_csv(pooled_path, index=False)

    cv_metrics = None
    project = folder.parent / "project" / "metrics.csv"
    if protocol == "paper" and project.exists():
        cv_metrics = read_metrics(project)

    comparisons_path = folder / "comparisons.csv"
    new = pd.DataFrame()
    if comparisons is not None:
        new = comparisons[comparisons["imputation_run_id"] == imputation_run_id]
    folder_comparisons = _merge_comparisons(comparisons_path, new, set(tests))
    if len(folder_comparisons) or comparisons_path.exists():
        folder_comparisons.to_csv(comparisons_path, index=False)

    summary = summarize(all_metrics, cv_metrics, all_pooled, targets)
    title = f"Forecast evaluation: {imputation_run_id}, {protocol} protocol"
    paths["summary_csv"], paths["summary_md"] = write_summary(
        summary, folder, title, folder_comparisons
    )

    top_path = folder / "top_misforecasts.csv"
    top = [
        top_misforecasts(pd.concat([frame for _, frame in parts]), k=10).assign(
            **dict(zip(GROUP, group))
        )
        for group, parts in tests.items()
        if group[1] == imputation_run_id
    ]
    if top:
        top = pd.concat(top, ignore_index=True)
        top = top[GROUP + [c for c in top.columns if c not in GROUP]]
        _merge_rows(top_path, top, GROUP, pd.read_csv).to_csv(top_path, index=False)
    if top_path.exists():
        paths["top_misforecasts"] = top_path
    return paths, summary


def _ledger_rows(tests: dict, metrics: pd.DataFrame, stamp: str) -> pd.DataFrame:
    rows = []
    for group, parts in tests.items():
        infos = [info for info, _ in parts]
        mine = metrics[
            np.logical_and.reduce(
                [(metrics[k] == v).to_numpy() for k, v in zip(GROUP, group)]
            )
        ]
        wape = mine[
            (mine["split"] == "test")
            & (mine["slice"] == "all")
            & (mine["metric"] == "wape")
        ]
        rows.append(
            {
                "scored_at_utc": stamp,
                **dict(zip(GROUP, group)),
                "seeds": " ".join(
                    str(seed) for seed in sorted({i.seed for i in infos})
                ),
                "n_tables": len(infos),
                "tables_sha256": combined_sha256([file_sha256(i.path) for i in infos]),
                "wape_mean": wape["value"].mean(),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    tables: list[Path],
    truth_path: str | Path = DEFAULT_TRUTH,
    slices_path: str | Path = DEFAULT_SLICES,
    point: str = "estimated_demand",
    quantiles_root: str | Path = DEFAULT_QUANTILES,
    protocol: str = "project",
    out_dir: str | Path = DEFAULT_OUT,
    baseline_model: str | None = None,
    ledger: bool = True,
    data_config: str | Path = DEFAULT_DATA_CONFIG,
    targets_path: str | Path = DEFAULT_TARGETS,
) -> dict:
    """Score forecast tables one by one and write the outputs under <out_dir>/<imputation_run_id>/<protocol>/.

    Returns the written paths keyed by imputation_run_id ("out_dirs", "metrics", "summary_csv", "summary_md",
    "top_misforecasts"), the ledger path (None if not written), the summary of every output folder written
    ("summary") and this call's baseline comparisons ("comparisons").
    """
    if point not in POINTS:
        raise ValueError(f"point must be one of {POINTS}, not {point!r}")
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, not {protocol!r}")
    infos = [table_info(resolve(path)) for path in tables]
    if not infos:
        raise ValueError("no forecast tables to evaluate")
    seen = {}
    for info in infos:
        part = (info.model_id, info.imputation_run_id, info.seed, info.fold, info.split)
        if part in seen:
            raise ValueError(
                f"{info.path} and {seen[part]} hold the same run part {part}"
            )
        seen[part] = info.path
    if baseline_model and not any(
        i.model_id == baseline_model and i.split == "test" for i in infos
    ):
        raise ValueError(
            f"no test table of the baseline model {baseline_model} among the tables"
        )

    splits_cfg = load_config(data_config)["splits"]
    targets = load_targets(targets_path)
    truth_path = resolve(truth_path)
    truth = load_truth(truth_path)
    slices = load_slices(resolve(slices_path), truth, truth_path)

    scales, metrics, pooled, tests = {}, [], [], {}
    for info in infos:
        table = read_table(info, point, quantiles_root)
        scored = score_table(
            table,
            truth,
            point=point,
            check_actuals=info.imputation_run_id == CENSORED_RUN,
        )
        scored = attach_slices(scored, slices)
        window = training_window(splits_cfg, info.fold, protocol)
        if (window.start, window.end) not in scales:
            scales[(window.start, window.end)] = series_scale(
                truth, window.start, window.end
            )
        meta = {
            "model_id": info.model_id,
            "imputation_run_id": info.imputation_run_id,
            "protocol": protocol,
            "point": point,
            "fold": info.fold,
            "seed": info.seed,
            "split": info.split,
        }
        metrics.append(
            metric_rows(
                scored,
                scales[(window.start, window.end)],
                meta,
                intervals=has_bounds(table),
                calibrated=has_bounds(table, calibrated=True),
            )
        )
        pooled.append(pooled_wape(scored).assign(feature=FEATURE, **meta))
        if info.split == "test":
            group = (info.model_id, info.imputation_run_id, protocol, point)
            tests.setdefault(group, []).append((info, scored[TEST_COLUMNS]))
        print(f"scored {info.path}: {len(scored):,} of {len(table):,} rows")
        del table, scored

    metrics = pd.concat(metrics, ignore_index=True)
    check_metric_keys(metrics)
    pooled = _pooled_frame(pd.concat(pooled, ignore_index=True))
    comparisons = _comparisons(tests, baseline_model) if baseline_model else None

    out_root = resolve(out_dir)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    result = {key: {} for key in FOLDER_OUTPUTS}
    summaries = []
    with _output_lock(out_root):
        for imputation_run_id in sorted(metrics["imputation_run_id"].unique()):
            paths, summary = _write_folder(
                out_root / imputation_run_id / protocol,
                metrics[metrics["imputation_run_id"] == imputation_run_id],
                pooled[pooled["imputation_run_id"] == imputation_run_id],
                comparisons,
                tests,
                targets,
            )
            for key, path in paths.items():
                result[key][imputation_run_id] = path
            summaries.append(summary)
        result["ledger"] = None
        if ledger and tests:
            result["ledger"] = append_ledger(
                out_root / "test_ledger.csv", _ledger_rows(tests, metrics, stamp)
            )
    result["summary"] = pd.concat(summaries, ignore_index=True)
    result["comparisons"] = comparisons

    for md_path in result["summary_md"].values():
        print(f"summary: {md_path}")
    print(short_table(result["summary"]))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        required=True,
        help="forecast table globs (** is recursive), relative to the repo root",
    )
    parser.add_argument(
        "--truth",
        default=DEFAULT_TRUTH,
        help="censored demand_daily: the raw-sales truth",
    )
    parser.add_argument(
        "--slices",
        default=DEFAULT_SLICES,
        help="sales slices, built from --truth if missing",
    )
    parser.add_argument("--point", choices=POINTS, default="estimated_demand")
    parser.add_argument(
        "--quantiles-root",
        default=DEFAULT_QUANTILES,
        help="run directories searched for quantiles_<split>.parquet (--point q500)",
    )
    parser.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default="project",
        help="paper: the final model was trained through the calibration week",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--baseline-model",
        help="model_id whose test WAPE every other model is compared against (paired bootstrap)",
    )
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        help="do not append to <out>/test_ledger.csv",
    )
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    args = parser.parse_args(argv)

    evaluate(
        expand_tables(args.tables),
        truth_path=args.truth,
        slices_path=args.slices,
        point=args.point,
        quantiles_root=args.quantiles_root,
        protocol=args.protocol,
        out_dir=args.out,
        baseline_model=args.baseline_model,
        ledger=not args.no_ledger,
        data_config=args.data_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
