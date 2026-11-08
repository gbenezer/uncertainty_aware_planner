"""Step 1: load FreshRetailNet-50K with Hugging Face `datasets` at a pinned revision and record its version."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import datasets
import pyarrow as pa
from huggingface_hub import hf_hub_download

from .config import resolve
from .outputs import file_sha256


def open_dataset(cfg: dict) -> datasets.DatasetDict:
    """Return the "train" and "eval" splits, from the Hub (pinned revision) or local parquet files."""
    source = cfg["source"]
    cache_dir = str(resolve(cfg["paths"]["cache_dir"]))
    if source["kind"] == "hub":
        return datasets.load_dataset(
            source["repo_id"], revision=source["revision"], cache_dir=cache_dir
        )
    if source["kind"] == "parquet":
        files = {
            split: str(resolve(path)) for split, path in source["data_files"].items()
        }
        return datasets.load_dataset("parquet", data_files=files, cache_dir=cache_dir)
    raise ValueError(f"unknown source kind {source['kind']!r}")


def to_arrow(dataset: datasets.Dataset) -> pa.Table:
    return dataset.with_format("arrow")[:]


def check_raw_files(cfg: dict) -> dict[str, dict]:
    """Hash the downloaded Hub files and stop unless they match `source.files` in configs/data.yaml.

    The files come from the Hugging Face cache, where `load_dataset` put them; with a pinned revision
    this needs no network. A different hash means different data, so results would not be comparable.
    """
    source = cfg["source"]
    pinned = {"repo_type": "dataset", "revision": source["revision"]}
    found = {}
    for name, expected in source["files"].items():
        path = Path(hf_hub_download(source["repo_id"], name, **pinned))
        found[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        if found[name] != expected:
            raise ValueError(
                f"{name} at revision {source['revision']} is {found[name]}, but configs/data.yaml "
                f"expects {expected}"
            )
    return found


def write_version_record(
    cfg: dict, dataset_dict: datasets.DatasetDict, files: dict[str, dict]
) -> Path:
    """Write data/raw/DATA_VERSION.json. The first download time is kept across re-runs."""
    path = resolve(cfg["paths"]["raw_dir"]) / "DATA_VERSION.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    source = cfg["source"]
    record = {
        "repo_id": source.get("repo_id"),
        "revision": source.get("revision"),
        "card_version": source.get("card_version"),
        "license": source.get("license"),
        "source_kind": source["kind"],
        "datasets_version": datasets.__version__,
        "downloaded_at_utc": previous.get("downloaded_at_utc", now),
        "last_loaded_at_utc": now,
        "num_rows": {split: ds.num_rows for split, ds in dataset_dict.items()},
        "files": files,
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
