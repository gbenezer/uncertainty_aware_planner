"""Writing data-stage outputs: parquet with contract types, plus a manifest for the run log."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATE_COLUMNS = ("date", "start_date", "end_date")


def write_parquet(df: pd.DataFrame, path: Path, overwrite: bool = True) -> None:
    """Write df as parquet; calendar-date columns are stored as dates (CONTRACTS.md 1)."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; versioned outputs are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    for column in DATE_COLUMNS:
        if column in table.column_names:
            index = table.column_names.index(column)
            table = table.set_column(
                index, column, pa.array(pd.to_datetime(df[column]).dt.date, pa.date32())
            )
    pq.write_table(table, path)


def write_manifest(
    output: Path, cfg: dict, config_path: str, seed: int | None, details: dict
) -> Path:
    """Write <output>.manifest.json: what produced this output, for pipeline/'s run log."""
    manifest = {
        "output": str(output),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config_path,
        "seed": seed,
        "dataset_revision": cfg["source"].get("revision"),
        **details,
    }
    path = output.with_name(output.name + ".manifest.json")
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path
