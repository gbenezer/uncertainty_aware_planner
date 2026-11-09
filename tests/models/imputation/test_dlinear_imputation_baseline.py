"""Tests for the DLinear_Imputation_Baseline wrapper.

PyPOTS' DLinear and Adam are replaced with mocks, so these tests are fast and
check only the wrapper's own logic (config validation, argument forwarding,
path handling, return values).

Adjust the import below to match the module name of the file under test.
Generated with Claude
"""
import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import uncertainty_aware_planner.models.imputation.dlinear_imputation_baseline as mod
from uncertainty_aware_planner.models.imputation.dlinear_imputation_baseline import DLinear_Imputation_Baseline


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg():
    return {
        "name": "DLinear_Imputation_Baseline",
        "n_steps": 24,
        "n_features": 5,
        "moving_avg_window_size": 7,
        "individual": False,
        "d_model": 32,
        "ORT_weight": 1.0,
        "MIT_weight": 0.5,
        "batch_size": 16,
        "epochs": 3,
        "patience": 2,
        "optimizer_parameters": {"lr": 1e-3, "weight_decay": 1e-5},
    }


@pytest.fixture
def mock_dlinear(monkeypatch):
    """Replace the DLinear class in the module under test."""
    m = MagicMock(name="DLinear")
    monkeypatch.setattr(mod, "DLinear", m)
    return m


@pytest.fixture
def mock_adam(monkeypatch):
    """Replace the Adam class in the module under test."""
    m = MagicMock(name="Adam")
    monkeypatch.setattr(mod, "Adam", m)
    return m


@pytest.fixture
def baseline(cfg, mock_dlinear, mock_adam):
    return DLinear_Imputation_Baseline(cfg)


@pytest.fixture
def inner(baseline):
    """The mocked DLinear instance held by the wrapper."""
    return baseline.model


# --------------------------------------------------------------------------- #
# __init__: config validation
# --------------------------------------------------------------------------- #
class TestInitValidation:
    def test_wrong_name_raises_value_error(self, cfg, mock_dlinear, mock_adam):
        cfg["name"] = "SomethingElse"
        with pytest.raises(ValueError, match="expected DLinear_Imputation_Baseline, got SomethingElse"):
            DLinear_Imputation_Baseline(cfg)

    def test_wrong_name_does_not_build_model(self, cfg, mock_dlinear, mock_adam):
        cfg["name"] = "nope"
        with pytest.raises(ValueError):
            DLinear_Imputation_Baseline(cfg)
        mock_dlinear.assert_not_called()
        mock_adam.assert_not_called()

    def test_missing_name_raises_key_error(self, cfg, mock_dlinear, mock_adam):
        del cfg["name"]
        with pytest.raises(KeyError):
            DLinear_Imputation_Baseline(cfg)

    @pytest.mark.parametrize(
        "key",
        [
            "n_steps",
            "n_features",
            "moving_avg_window_size",
            "individual",
            "d_model",
            "ORT_weight",
            "MIT_weight",
            "batch_size",
            "epochs",
            "patience",
            "optimizer_parameters",
        ],
    )
    def test_missing_required_key_raises_key_error(self, cfg, mock_dlinear, mock_adam, key):
        del cfg[key]
        with pytest.raises(KeyError, match=key):
            DLinear_Imputation_Baseline(cfg)

    def test_config_is_not_mutated(self, cfg, mock_dlinear, mock_adam):
        before = copy.deepcopy(cfg)
        DLinear_Imputation_Baseline(cfg)
        assert cfg == before


# --------------------------------------------------------------------------- #
# __init__: model construction
# --------------------------------------------------------------------------- #
class TestInitConstruction:
    def test_model_attribute_is_dlinear_instance(self, baseline, mock_dlinear):
        assert baseline.model is mock_dlinear.return_value

    def test_dlinear_called_once(self, baseline, mock_dlinear):
        mock_dlinear.assert_called_once()

    def test_config_values_forwarded_to_dlinear(self, cfg, baseline, mock_dlinear):
        kwargs = mock_dlinear.call_args.kwargs
        for key in (
            "n_steps",
            "n_features",
            "moving_avg_window_size",
            "individual",
            "d_model",
            "ORT_weight",
            "MIT_weight",
            "batch_size",
            "epochs",
            "patience",
        ):
            assert kwargs[key] == cfg[key], key

    def test_fixed_arguments(self, baseline, mock_dlinear):
        kwargs = mock_dlinear.call_args.kwargs
        assert kwargs["num_workers"] == 0
        assert kwargs["model_saving_strategy"] == "best"
        assert kwargs["verbose"] is True

    def test_training_loss_and_validation_metric_left_at_defaults(self, baseline, mock_dlinear):
        kwargs = mock_dlinear.call_args.kwargs
        assert "training_loss" not in kwargs
        assert "validation_metric" not in kwargs

    def test_optimizer_built_from_optimizer_parameters(self, cfg, baseline, mock_adam, mock_dlinear):
        mock_adam.assert_called_once_with(**cfg["optimizer_parameters"])
        assert mock_dlinear.call_args.kwargs["optimizer"] is mock_adam.return_value

    def test_empty_optimizer_parameters_allowed(self, cfg, mock_dlinear, mock_adam):
        cfg["optimizer_parameters"] = {}
        DLinear_Imputation_Baseline(cfg)
        mock_adam.assert_called_once_with()


# --------------------------------------------------------------------------- #
# __init__: saving_path handling
# --------------------------------------------------------------------------- #
class TestSavingPath:
    def test_none_saving_path(self, cfg, mock_dlinear, mock_adam):
        b = DLinear_Imputation_Baseline(cfg)
        assert b.saving_path is None
        assert mock_dlinear.call_args.kwargs["saving_path"] is None

    def test_str_saving_path_converted_to_path(self, cfg, mock_dlinear, mock_adam, tmp_path):
        b = DLinear_Imputation_Baseline(cfg, saving_path=str(tmp_path))
        assert isinstance(b.saving_path, Path)
        assert b.saving_path == tmp_path

    def test_path_saving_path_kept(self, cfg, mock_dlinear, mock_adam, tmp_path):
        b = DLinear_Imputation_Baseline(cfg, saving_path=tmp_path)
        assert b.saving_path == tmp_path

    def test_saving_path_passed_to_dlinear_as_str(self, cfg, mock_dlinear, mock_adam, tmp_path):
        DLinear_Imputation_Baseline(cfg, saving_path=tmp_path)
        passed = mock_dlinear.call_args.kwargs["saving_path"]
        assert isinstance(passed, str)
        assert passed == str(tmp_path)

    def test_init_does_not_create_saving_path(self, cfg, mock_dlinear, mock_adam, tmp_path):
        target = tmp_path / "not_created_yet"
        DLinear_Imputation_Baseline(cfg, saving_path=target)
        assert not target.exists()


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
class TestFit:
    def test_forwards_train_and_val_sets(self, baseline, inner):
        train, val = {"X": np.zeros((2, 3, 4))}, {"X": np.ones((2, 3, 4))}
        baseline.fit(train, val)
        inner.fit.assert_called_once_with(train, val)

    def test_val_set_defaults_to_none(self, baseline, inner):
        train = {"X": np.zeros((2, 3, 4))}
        baseline.fit(train)
        inner.fit.assert_called_once_with(train, None)

    def test_accepts_h5_file_paths(self, baseline, inner):
        baseline.fit("train.h5", "val.h5")
        inner.fit.assert_called_once_with("train.h5", "val.h5")

    def test_returns_self_for_chaining(self, baseline):
        assert baseline.fit({"X": np.zeros((1, 1, 1))}) is baseline

    def test_fit_exceptions_propagate(self, baseline, inner):
        inner.fit.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            baseline.fit({"X": np.zeros((1, 1, 1))})


# --------------------------------------------------------------------------- #
# impute
# --------------------------------------------------------------------------- #
class TestImpute:
    def test_returns_model_output_unchanged(self, baseline, inner):
        expected = np.random.default_rng(0).normal(size=(4, 24, 5))
        inner.impute.return_value = expected
        out = baseline.impute({"X": np.zeros((4, 24, 5))})
        assert out is expected

    def test_forwards_test_set(self, baseline, inner):
        test_set = {"X": np.zeros((4, 24, 5))}
        baseline.impute(test_set)
        inner.impute.assert_called_once_with(test_set)

    def test_accepts_file_path(self, baseline, inner):
        baseline.impute("test.h5")
        inner.impute.assert_called_once_with("test.h5")

    def test_impute_exceptions_propagate(self, baseline, inner):
        inner.impute.side_effect = ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            baseline.impute({"X": np.zeros((1, 1, 1))})


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
class TestSave:
    def test_calls_model_save_with_expected_file(self, baseline, inner, tmp_path):
        baseline.save(tmp_path)
        inner.save.assert_called_once_with(str(tmp_path / "dlinear_imputation_baseline.pypots"), overwrite=False)

    def test_overwrite_flag_forwarded(self, baseline, inner, tmp_path):
        baseline.save(tmp_path, overwrite=True)
        assert inner.save.call_args.kwargs["overwrite"] is True

    def test_creates_directory(self, baseline, tmp_path):
        target = tmp_path / "does_not_exist"
        baseline.save(target)
        assert target.is_dir()

    def test_creates_nested_directories(self, baseline, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        baseline.save(target)
        assert target.is_dir()

    def test_existing_directory_is_ok(self, baseline, tmp_path):
        baseline.save(tmp_path)
        baseline.save(tmp_path)  # second call must not raise
        assert inner_calls(baseline) == 2

    def test_accepts_str_directory(self, baseline, inner, tmp_path):
        baseline.save(str(tmp_path))
        inner.save.assert_called_once_with(str(tmp_path / "dlinear_imputation_baseline.pypots"), overwrite=False)

    def test_path_passed_to_model_is_str(self, baseline, inner, tmp_path):
        baseline.save(tmp_path)
        assert isinstance(inner.save.call_args.args[0], str)

    def test_save_exceptions_propagate(self, baseline, inner, tmp_path):
        inner.save.side_effect = FileExistsError("exists")
        with pytest.raises(FileExistsError):
            baseline.save(tmp_path)


def inner_calls(baseline):
    return baseline.model.save.call_count


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
class TestLoad:
    def test_calls_model_load_with_expected_file(self, baseline, inner, tmp_path):
        baseline.load(tmp_path)
        inner.load.assert_called_once_with(str(tmp_path / "dlinear_imputation_baseline.pypots"))

    def test_accepts_str_directory(self, baseline, inner, tmp_path):
        baseline.load(str(tmp_path))
        inner.load.assert_called_once_with(str(tmp_path / "dlinear_imputation_baseline.pypots"))

    def test_does_not_create_directory(self, baseline, tmp_path):
        target = tmp_path / "missing"
        baseline.load(target)
        assert not target.exists()

    def test_returns_none(self, baseline, tmp_path):
        assert baseline.load(tmp_path) is None

    def test_load_exceptions_propagate(self, baseline, inner, tmp_path):
        inner.load.side_effect = FileNotFoundError("no such file")
        with pytest.raises(FileNotFoundError):
            baseline.load(tmp_path)


# --------------------------------------------------------------------------- #
# save / load consistency
# --------------------------------------------------------------------------- #
def test_save_and_load_use_same_filename(baseline, inner, tmp_path):
    baseline.save(tmp_path)
    baseline.load(tmp_path)
    saved = inner.save.call_args.args[0]
    loaded = inner.load.call_args.args[0]
    assert saved == loaded