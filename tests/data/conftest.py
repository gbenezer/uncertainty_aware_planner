import pytest

from fixtures.synthetic_raw_series import fixture_config, make_raw_series
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.masks import make_eval_masks


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    return fixture_config(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def cube(cfg):
    return load_cube(cfg)


@pytest.fixture(scope="session")
def raw_tables():
    return make_raw_series()


@pytest.fixture(scope="session")
def masks(cube, cfg):
    return make_eval_masks(cube, cfg["splits"])
