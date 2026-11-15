import numpy as np

from uncertainty_aware_planner.data.config import load_yaml
from uncertainty_aware_planner.data.cube import load_cube
from uncertainty_aware_planner.data.masks import make_eval_masks
from uncertainty_aware_planner.pipeline.baseline_smoke import (
    DEFAULT_CONFIG,
    check_data,
    data_cfg_for_fixtures,
    train_and_score,
)


def test_fixture_hashes_match_the_smoke_config():
    smoke = load_yaml(DEFAULT_CONFIG)
    found = check_data(smoke["source"])
    assert set(found) == set(smoke["source"]["files"])


def test_same_seed_gives_the_same_baseline_number(tmp_path):
    smoke = load_yaml(DEFAULT_CONFIG) | {"seeds": [86400], "folds": [1]}
    cfg = data_cfg_for_fixtures(smoke, tmp_path)
    cube = load_cube(cfg)
    masks = make_eval_masks(cube, cfg["splits"])
    model_cfg = load_yaml(smoke["model_config"])
    first = train_and_score(cube, cfg, masks, model_cfg, smoke)
    second = train_and_score(cube, cfg, masks, model_cfg, smoke)
    assert first == second
    assert [r["variant"] for r in first] == ["not_leaked", "leaked"]
    assert all(np.isfinite(r["wape"]) and r["wape"] > 0 for r in first)
