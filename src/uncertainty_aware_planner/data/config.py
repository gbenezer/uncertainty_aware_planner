"""Load the data-stage YAML configs. Relative paths in configs are resolved against the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load data.yaml and attach the parsed splits.yaml it points to under the key "splits"."""
    cfg = load_yaml(path)
    cfg["splits"] = load_yaml(cfg["splits_config"])
    return cfg
