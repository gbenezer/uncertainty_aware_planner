"""The data stages run end to end on the synthetic fixtures through the command line."""

import json
from datetime import date

import pandas as pd
import pytest
import yaml

from tests.data.helpers import fake_imputed_hourly
from uncertainty_aware_planner.data.__main__ import main
from uncertainty_aware_planner.data.config import REPO_ROOT


@pytest.fixture
def config_path(cfg, tmp_path):
    data_cfg = {key: value for key, value in cfg.items() if key != "splits"}
    data_cfg["splits_config"] = str(REPO_ROOT / "configs" / "splits.yaml")
    data_cfg["paths"] = {
        name: str(tmp_path / name) for name in ("cache_dir", "raw_dir", "processed_dir")
    }
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(data_cfg))
    return path


def test_stages_run_end_to_end(config_path, tmp_path, cube, capsys):
    processed = tmp_path / "processed_dir"
    args = ["--config", str(config_path)]
    main(["load", *args])
    main(["splits", *args])
    assert len(pd.read_parquet(processed / "splits.parquet")) == 15
    main(["masks", *args])
    masks = pd.read_parquet(processed / "eval_masks.parquet")
    assert len(masks) > 0

    imputed_path = tmp_path / "imputed_hourly.parquet"
    fake_imputed_hourly(
        cube, date(2024, 3, 28), date(2024, 6, 25), masks=masks
    ).to_parquet(imputed_path)
    main(
        [
            "daily",
            "--imputed",
            str(imputed_path),
            "--run-id",
            "imp-dlinear-quantile-1024",
            *args,
        ]
    )
    output = processed / "demand_daily" / "imp-dlinear-quantile-1024.parquet"
    demand = pd.read_parquet(output)
    assert len(demand) == 6 * 97
    assert isinstance(demand["date"].iloc[0], date)
    manifest = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    assert manifest["rows"] == 6 * 97 and manifest["edit_ids"] == []
    with pytest.raises(
        FileExistsError
    ):  # versioned demand_daily files are never overwritten
        main(
            [
                "daily",
                "--imputed",
                str(imputed_path),
                "--run-id",
                "imp-dlinear-quantile-1024",
                *args,
            ]
        )
