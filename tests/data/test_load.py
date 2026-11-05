import pytest

from fixtures.synthetic_raw_series import FIXTURE_DIR
from uncertainty_aware_planner.data import load
from uncertainty_aware_planner.data.config import load_config
from uncertainty_aware_planner.data.outputs import file_sha256

FILES = {
    "data/train.parquet": FIXTURE_DIR / "train.parquet",
    "data/eval.parquet": FIXTURE_DIR / "eval.parquet",
}


def hub_config(cfg, files):
    return {**cfg, "source": {"repo_id": "x", "revision": "y", "files": files}}


@pytest.fixture
def cached_hub_files(monkeypatch):
    # Stand-in for the Hugging Face cache: the fixture files play the downloaded Hub files.
    monkeypatch.setattr(load, "hf_hub_download", lambda repo, name, **_: FILES[name])


def test_raw_files_matching_the_config_are_recorded(cfg, cached_hub_files):
    expected = {
        name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for name, path in FILES.items()
    }
    assert load.check_raw_files(hub_config(cfg, expected)) == expected


def test_a_raw_file_with_another_hash_stops_the_load(cfg, cached_hub_files):
    expected = {
        name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for name, path in FILES.items()
    }
    expected["data/eval.parquet"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data/eval.parquet"):
        load.check_raw_files(hub_config(cfg, expected))


def test_the_committed_config_pins_both_raw_files():
    files = load_config()["source"]["files"]
    assert set(files) == {"data/train.parquet", "data/eval.parquet"}
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in files.values())
