"""Forecast evaluation summary: test-week WAPE and WPE over seeds, the fold-CV spread, and the dataset paper's
published numbers (configs/evaluation/published_targets.yaml) with the acceptance verdicts (Dev Contract A.7.2).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.evaluation.forecast_metrics import SLICES

DEFAULT_TARGETS = "configs/evaluation/published_targets.yaml"
GROUP = ["model_id", "imputation_run_id", "protocol", "point"]
VERDICTS = ["pass", "beats_by_1sd", "a72_literal"]
SUMMARY_COLUMNS = GROUP + [
    "slice",
    "n_seeds",
    "test_wape_mean",
    "test_wape_sd",
    "test_wpe_mean",
    "test_wpe_sd",
    "test_paper_wape_mean",
    "cv_n",
    "cv_wape_mean",
    "cv_wape_sd",
    "published_wape",
    "published_wpe",
    "pass",
    "beats_by_1sd",
    "improvement_sd",
    "a72_literal",
    "test_mase",
    "test_picp95",
    "test_mpiw95",
    "test_picp95_cal",
    "test_mpiw95_cal",
]
LEDGER_COLUMNS = [
    "scored_at_utc",
    "model_id",
    "imputation_run_id",
    "protocol",
    "point",
    "seeds",
    "n_tables",
    "tables_sha256",
    "wape_mean",
]
NOTES = """\
- **Truth:** raw recorded sales (`demand_total` of the censored demand_daily) for every model, whatever demand
  history it was trained on. Only target days with no stockout between 06:00 and 22:00 are scored.
- **WAPE** = sum|F - A| / sum|A| and **WPE** = sum(F - A) / sum|A| over all scored rows (the literature
  definition, Dev Contract A.4); each row is one store-product-day, so no errors are netted before the
  absolute value. WPE sign as in the dataset paper: positive = over-forecast, negative = under-forecast.
  **Paper-style WAPE** is the dataset paper's own scoring (the ratio per target day, averaged over the days),
  shown only to compare with its published numbers.
- **Test:** mean ± SD over seeds (number of seeds). **Fold CV:** mean ± SD over all (fold, seed) validation
  runs (number of runs); SD_cv is that SD.
- **Published:** Wang et al. 2025 (arXiv:2505.16319), Table 3. **Pass (A.7.2):** test WAPE <= published +
  SD_cv, with SD_cv from the >= 3-fold cross-validation (Dev Contract, Appendix A.7, item 2).
  **>= 1 SD better:** test WAPE <= published - SD_cv, an optional team goal (not a contract term).
  **Improvement:** (published - test WAPE) / SD_cv. **A.7.2 literal:** |fold CV mean - published| <= SD_cv.
- **PICP / MPIW:** the team functions in `evaluation/uncertainty_metrics.py`: rows with both bounds,
  degenerate intervals (lower = upper, e.g. [0, 0]) left out.
- Metric rows follow `contracts/CONTRACTS.md` §2.10, plus imputation_run_id, protocol, point and n_skipped.
- WAPE, WPE and PICP in percent; MASE and MPIW (normalized sales units) as plain numbers."""
BOOTSTRAP_NOTE = """\
Test week, all series; forecasts averaged over seeds first. Each draw resamples series with replacement, the
same draw for both models, and recomputes WAPE over all rows. Delta = model WAPE - baseline WAPE (negative
favours the model), with its 95% bootstrap interval."""


def load_targets(path: str | Path = DEFAULT_TARGETS) -> dict:
    return load_yaml(path)


def _lookup(mapping: dict, identifier: str) -> str | None:
    keys = [
        key for key in mapping if identifier == key or identifier.startswith(key + "-")
    ]
    return mapping[max(keys, key=len)] if keys else None


def published_target(
    targets: dict, model_id: str, imputation_run_id: str
) -> dict | None:
    """Published {slice: {wape, wpe}} as fractions for this model and training data, or None."""
    family = _lookup(targets["model_families"], model_id)
    data = _lookup(targets["training_data"], imputation_run_id)
    table = targets["targets"].get(family, {}).get(data) if family and data else None
    if table is None:
        return None
    return {
        name: {metric: value / 100 for metric, value in values.items()}
        for name, values in table.items()
    }


def _select(frame: pd.DataFrame, values: dict) -> pd.DataFrame:
    mask = np.logical_and.reduce(
        [(frame[key] == value).to_numpy() for key, value in values.items()]
    )
    return frame[mask]


def _values(
    frame: pd.DataFrame, split: str, metric: str, slice_: str, calibrated=False
):
    rows = frame[
        (frame["split"] == split)
        & (frame["metric"] == metric)
        & (frame["slice"] == slice_)
        & (frame["calibrated"] == calibrated)
    ]
    values = rows["value"].to_numpy(dtype="float64")
    return values[~np.isnan(values)]


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else np.nan


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else np.nan


def verdicts(test_mean: float, cv_mean: float, cv_sd: float, published: float) -> dict:
    """Pass, >= 1 SD better, improvement in SD_cv units and the literal A.7.2 check; NA when an input is missing."""
    test_known = bool(np.isfinite([test_mean, cv_sd, published]).all())
    cv_known = bool(np.isfinite([cv_mean, cv_sd, published]).all())
    return {
        "pass": test_mean <= published + cv_sd if test_known else pd.NA,
        "beats_by_1sd": test_mean <= published - cv_sd if test_known else pd.NA,
        "improvement_sd": (published - test_mean) / cv_sd
        if test_known and cv_sd > 0
        else np.nan,
        "a72_literal": abs(cv_mean - published) <= cv_sd if cv_known else pd.NA,
    }


def summarize(
    metrics: pd.DataFrame,
    cv_metrics: pd.DataFrame | None = None,
    paper: pd.DataFrame | None = None,
    targets: dict | None = None,
) -> pd.DataFrame:
    """One row per (model_id, imputation_run_id, protocol, point, slice), values as fractions.

    Test values are means (and SDs) over seeds of the uncalibrated test rows. Fold CV is the mean and SD
    (SD_cv) of WAPE over all (fold, seed) val rows of the same group, or, when `cv_metrics` is given (the paper
    protocol borrowing the project folds), of its rows with the same model_id, imputation_run_id and point.
    """
    rows = []
    groups = metrics[GROUP].drop_duplicates().sort_values(GROUP)
    for values in groups.itertuples(index=False):
        group = dict(zip(GROUP, values))
        mine = _select(metrics, group)
        cv = mine
        if cv_metrics is not None:
            same_run = {
                key: group[key] for key in ("model_id", "imputation_run_id", "point")
            }
            cv = _select(cv_metrics, same_run)
        runs = _select(paper, group) if paper is not None else None
        target = None
        if targets is not None:
            target = published_target(
                targets, group["model_id"], group["imputation_run_id"]
            )
        for name in SLICES:
            wape = _values(mine, "test", "wape", name)
            wpe = _values(mine, "test", "wpe", name)
            cv_wape = _values(cv, "val", "wape", name)
            published = (target or {}).get(name, {})
            row = {
                **group,
                "slice": name,
                "n_seeds": len(wape),
                "test_wape_mean": _mean(wape),
                "test_wape_sd": _sd(wape),
                "test_wpe_mean": _mean(wpe),
                "test_wpe_sd": _sd(wpe),
                "test_paper_wape_mean": np.nan,
                "cv_n": len(cv_wape),
                "cv_wape_mean": _mean(cv_wape),
                "cv_wape_sd": _sd(cv_wape),
                "published_wape": published.get("wape", np.nan),
                "published_wpe": published.get("wpe", np.nan),
            }
            if runs is not None:
                test = runs[(runs["split"] == "test") & (runs["slice"] == name)]
                row["test_paper_wape_mean"] = _mean(test["value"].dropna().to_numpy())
            row.update(
                verdicts(
                    row["test_wape_mean"],
                    row["cv_wape_mean"],
                    row["cv_wape_sd"],
                    row["published_wape"],
                )
            )
            for metric in ("mase", "picp95", "mpiw95"):
                row[f"test_{metric}"] = _mean(_values(mine, "test", metric, name))
            for metric in ("picp95", "mpiw95"):
                row[f"test_{metric}_cal"] = _mean(
                    _values(mine, "test", metric, name, True)
                )
            rows.append(row)
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    for column in VERDICTS:
        summary[column] = summary[column].astype("boolean")
    return summary


def _number(value, scale: float = 100, digits: int = 2) -> str:
    return f"{scale * value:.{digits}f}" if pd.notna(value) else "n/a"


def _spread(mean, sd, n) -> str:
    if pd.isna(mean):
        return "n/a"
    text = _number(mean)
    if pd.notna(sd):
        text += f" ± {_number(sd)}"
    return f"{text} ({n})"


def _flag(value) -> str:
    return "n/a" if pd.isna(value) else ("yes" if value else "no")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def summary_markdown(
    summary: pd.DataFrame, title: str, comparisons: pd.DataFrame | None = None
) -> str:
    """Readable summary.md: one accuracy table and one interval table per (model_id, point)."""
    lines = [f"# {title}", "", NOTES]
    if (summary["protocol"] == "paper").any():
        lines.append(
            "- Paper protocol: fold CV comes from the project-protocol folds of the same model and data."
        )
    accuracy = ["Slice", "Test WAPE", "Test WPE", "Paper-style WAPE", "Fold CV WAPE"]
    accuracy += [
        "Published WAPE / WPE",
        "Pass",
        ">= 1 SD better (optional)",
        "Improvement (SD_cv)",
        "A.7.2 literal",
    ]
    intervals = ["Slice", "Test MASE", "Test PICP95", "Test MPIW95"]
    intervals += ["Test PICP95 (cal)", "Test MPIW95 (cal)"]
    for (model_id, imputation_run_id, protocol, point), rows in summary.groupby(
        GROUP, sort=False
    ):
        records = rows.to_dict("records")
        lines += [
            "",
            f"## {model_id} ({imputation_run_id}, {protocol}, point {point})",
            "",
        ]
        lines += _table(
            accuracy,
            [
                [
                    r["slice"],
                    _spread(r["test_wape_mean"], r["test_wape_sd"], r["n_seeds"]),
                    _spread(r["test_wpe_mean"], r["test_wpe_sd"], r["n_seeds"]),
                    _number(r["test_paper_wape_mean"]),
                    _spread(r["cv_wape_mean"], r["cv_wape_sd"], r["cv_n"]),
                    f"{_number(r['published_wape'])} / {_number(r['published_wpe'])}",
                    _flag(r["pass"]),
                    _flag(r["beats_by_1sd"]),
                    _number(r["improvement_sd"], scale=1),
                    _flag(r["a72_literal"]),
                ]
                for r in records
            ],
        )
        lines.append("")
        lines += _table(
            intervals,
            [
                [
                    r["slice"],
                    _number(r["test_mase"], scale=1, digits=3),
                    _number(r["test_picp95"]),
                    _number(r["test_mpiw95"], scale=1, digits=3),
                    _number(r["test_picp95_cal"]),
                    _number(r["test_mpiw95_cal"], scale=1, digits=3),
                ]
                for r in records
            ],
        )
    if comparisons is not None and len(comparisons):
        lines += [
            "",
            "## Paired bootstrap against the baseline",
            "",
            BOOTSTRAP_NOTE,
            "",
        ]
        header = [
            "Model",
            "Imputation run",
            "Point",
            "Baseline",
            "Baseline imputation run",
        ]
        header += ["WAPE", "Baseline WAPE", "Delta", "95% CI", "Series", "Draws"]
        lines += _table(
            header,
            [
                [
                    r["model_id"],
                    r["imputation_run_id"],
                    r["point"],
                    r["baseline_model_id"],
                    r["baseline_imputation_run_id"],
                    _number(r["wape_a"]),
                    _number(r["wape_b"]),
                    _number(r["delta"]),
                    f"[{_number(r['ci_low'])}, {_number(r['ci_high'])}]",
                    str(r["n_series"]),
                    str(r["n_boot"]),
                ]
                for r in comparisons.to_dict("records")
            ],
        )
    return "\n".join(lines) + "\n"


def short_table(summary: pd.DataFrame) -> str:
    """The 'all' slice of a summary as plain text (percent), for the command line."""
    records = summary[summary["slice"] == "all"].to_dict("records")
    view = pd.DataFrame(
        {
            "model_id": [r["model_id"] for r in records],
            "imputation_run_id": [r["imputation_run_id"] for r in records],
            "point": [r["point"] for r in records],
            "test WAPE %": [
                _spread(r["test_wape_mean"], r["test_wape_sd"], r["n_seeds"])
                for r in records
            ],
            "test WPE %": [
                _spread(r["test_wpe_mean"], r["test_wpe_sd"], r["n_seeds"])
                for r in records
            ],
            "fold CV WAPE %": [
                _spread(r["cv_wape_mean"], r["cv_wape_sd"], r["cv_n"]) for r in records
            ],
            "published WAPE %": [_number(r["published_wape"]) for r in records],
            "pass": [_flag(r["pass"]) for r in records],
            "A.7.2 literal": [_flag(r["a72_literal"]) for r in records],
        }
    )
    return view.to_string(index=False)


def write_summary(
    summary: pd.DataFrame,
    out_dir: Path,
    title: str,
    comparisons: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Write summary.csv (fractions) and summary.md (percent) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / "summary.csv", out_dir / "summary.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(summary_markdown(summary, title, comparisons))
    return csv_path, md_path


def combined_sha256(hashes: list[str]) -> str:
    """One hash for a set of files: sha256 of their sorted, concatenated sha256 hex digests."""
    return hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest()


def append_ledger(path: Path, rows: pd.DataFrame) -> Path:
    """Append test-week scoring events to the ledger CSV, writing the header when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows[LEDGER_COLUMNS].to_csv(path, mode="a", header=not path.exists(), index=False)
    return path
