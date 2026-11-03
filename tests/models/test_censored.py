import numpy as np

from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.daily import build_demand_daily
from uncertainty_aware_planner.pipeline.censored import censored_imputed_hourly


def test_censored_run_keeps_recorded_sales(cfg):
    cube = load_cube(cfg)
    daily, _ = build_demand_daily(
        cube, censored_imputed_hourly(cube), "imp-none-censored", cfg["splits"]
    )
    assert daily["demand_total"].notna().all()
    np.testing.assert_allclose(
        daily["demand_total"].to_numpy(), cube.sales.sum(axis=-1).reshape(-1)
    )
