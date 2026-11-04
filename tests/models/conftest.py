from datetime import date

import pytest

from fixtures.synthetic_raw_series import fixture_config
from tests.data.helpers import fake_imputed_hourly
from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.daily import build_demand_daily
from uncertainty_aware_planner.pipeline.forecast import smoke_config


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    return fixture_config(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def demand(cfg):
    """Fixture stand-in for demand_daily (data README, Step 5): every stockout filled with 0.5."""
    cube = load_cube(cfg)
    imputed = fake_imputed_hourly(cube, date(2024, 3, 28), date(2024, 6, 25))
    daily, _ = build_demand_daily(cube, imputed, "imp-fixture-0", cfg["splits"])
    return daily


@pytest.fixture
def tft_cfg(tmp_path):
    """The committed TFT config with its smoke overrides, writing under tmp_path."""
    model_cfg = smoke_config(
        load_yaml("configs/forecasting_models/temporal_fusion_transformer.yaml")
    )
    model_cfg["output_dir"] = str(tmp_path / "forecast")
    model_cfg["run_dir"] = str(tmp_path / "runs")
    return model_cfg
