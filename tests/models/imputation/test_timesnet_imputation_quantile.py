"""Tests for the TimesNet_Imputation_Quantile wrapper.

PyPOTS' TimesNet and AdamW (and, for most tests, QuantileLoss) are replaced
with mocks, so these tests are fast and check only the wrapper's own logic:
config validation, one-model-per-quantile construction, filename tagging,
argument forwarding, quantile stacking/sorting, and save/load paths.

Generated with Claude
"""
import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import uncertainty_aware_planner.models.imputation.timesnet_imputation_quantile as mod
from uncertainty_aware_planner.models.imputation.timesnet_imputation_quantile import TimesNet_Imputation_Quantile

N, T, F = 4, 24, 5  # samples, steps, features used for impute tests


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg():
    return {
        "name": "TimesNet_Imputation_Quantile",
        "quantiles": [0.9, 0.1, 0.5],  # deliberately unsorted
        "n_steps": T,
        "n_features": F,
        "n_layers": 2,
        "top_k": 3,
        "d_model": 32,
        "d_ffn": 64,
        "n_kernels": 4,
        "dropout": 0.1,
        "apply_nonstationary_norm": False,
        "batch_size": 16,
        "epochs": 3,
        "patience": 2,
        "optimizer_parameters": {"lr": 1e-3, "weight_decay": 1e-5},
    }


@pytest.fixture
def mock_timesnet(monkeypatch):
    """TimesNet class mock that returns a *distinct* MagicMock per call."""
    m = MagicMock(name="TimesNet", side_effect=lambda **kw: MagicMock(name="TimesNet_instance"))
    monkeypatch.setattr(mod, "TimesNet", m)
    return m


@pytest.fixture
def mock_adamw(monkeypatch):
    """AdamW class mock that returns a distinct object per call."""
    created = []

    def _make(**kw):
        inst = MagicMock(name="AdamW_instance")
        created.append(inst)
        return inst

    m = MagicMock(name="AdamW", side_effect=_make)
    m.created = created  # instances in creation order
    monkeypatch.setattr(mod, "AdamW", m)
    return m


@pytest.fixture
def mock_qloss(monkeypatch):
    """QuantileLoss class mock; each instance records the quantile it got."""

    def _make(quantile):
        inst = MagicMock(name=f"QuantileLoss_{quantile}")
        inst.quantile = quantile
        return inst

    m = MagicMock(name="QuantileLoss", side_effect=_make)
    monkeypatch.setattr(mod, "QuantileLoss", m)
    return m


@pytest.fixture
def model(cfg, mock_timesnet, mock_adamw, mock_qloss):
    return TimesNet_Imputation_Quantile(cfg)


def timesnet_kwargs_by_quantile(mock_timesnet, quantiles):
    """TimesNet is constructed once per quantile in sorted order."""
    return dict(zip(quantiles, [c.kwargs for c in mock_timesnet.call_args_list]))


# --------------------------------------------------------------------------- #
# _tag
# --------------------------------------------------------------------------- #
class TestTag:
    @pytest.mark.parametrize(
        "q, expected",
        [
            (0.1, "q100"),
            (0.5, "q500"),
            (0.9, "q900"),
            (0.05, "q050"),
            (0.01, "q010"),
            (0.025, "q025"),
            (0.975, "q975"),
            (0.999, "q999"),
        ],
    )
    def test_tag_format(self, q, expected):
        assert TimesNet_Imputation_Quantile._tag(q) == expected

    def test_tags_unique_for_typical_quantiles(self):
        qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        tags = [TimesNet_Imputation_Quantile._tag(q) for q in qs]
        assert len(set(tags)) == len(tags)

    def test_tag_is_float_rounding_safe(self):
        # 0.1 * 1000 == 100.00000000000001 in floating point
        assert TimesNet_Imputation_Quantile._tag(0.1) == "q100"
        assert TimesNet_Imputation_Quantile._tag(0.3) == "q300"
        assert TimesNet_Imputation_Quantile._tag(0.7) == "q700"


# --------------------------------------------------------------------------- #
# __init__: config validation
# --------------------------------------------------------------------------- #
class TestInitValidation:
    def test_wrong_name_raises_value_error(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["name"] = "TimesNet_Baseline"
        with pytest.raises(ValueError, match="expected TimesNet_Imputation_Quantile, got TimesNet_Baseline"):
            TimesNet_Imputation_Quantile(cfg)

    def test_wrong_name_builds_nothing(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["name"] = "nope"
        with pytest.raises(ValueError):
            TimesNet_Imputation_Quantile(cfg)
        mock_timesnet.assert_not_called()
        mock_adamw.assert_not_called()
        mock_qloss.assert_not_called()

    def test_missing_name_raises_key_error(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        del cfg["name"]
        with pytest.raises(KeyError):
            TimesNet_Imputation_Quantile(cfg)

    @pytest.mark.parametrize(
        "key",
        [
            "quantiles",
            "n_steps",
            "n_features",
            "n_layers",
            "top_k",
            "d_model",
            "d_ffn",
            "n_kernels",
            "dropout",
            "apply_nonstationary_norm",
            "batch_size",
            "epochs",
            "patience",
            "optimizer_parameters",
        ],
    )
    def test_missing_required_key_raises_key_error(self, cfg, mock_timesnet, mock_adamw, mock_qloss, key):
        del cfg[key]
        with pytest.raises(KeyError, match=key):
            TimesNet_Imputation_Quantile(cfg)

    def test_config_is_not_mutated(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        before = copy.deepcopy(cfg)
        TimesNet_Imputation_Quantile(cfg)
        assert cfg == before  # in particular, cfg["quantiles"] is not sorted in place


# --------------------------------------------------------------------------- #
# __init__: quantile handling and per-quantile models
# --------------------------------------------------------------------------- #
class TestQuantileModels:
    def test_quantiles_sorted_and_stored_as_tuple(self, model):
        assert model.quantiles == (0.1, 0.5, 0.9)
        assert isinstance(model.quantiles, tuple)

    def test_one_model_per_quantile(self, model, mock_timesnet):
        assert mock_timesnet.call_count == 3
        assert set(model.models.keys()) == {0.1, 0.5, 0.9}

    def test_models_dict_in_sorted_order(self, model):
        assert list(model.models.keys()) == [0.1, 0.5, 0.9]

    def test_models_are_distinct_instances(self, model):
        instances = list(model.models.values())
        assert len({id(m) for m in instances}) == len(instances)

    def test_single_quantile(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["quantiles"] = [0.5]
        m = TimesNet_Imputation_Quantile(cfg)
        assert m.quantiles == (0.5,)
        assert mock_timesnet.call_count == 1

    def test_empty_quantiles_builds_no_models(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["quantiles"] = []
        m = TimesNet_Imputation_Quantile(cfg)
        assert m.models == {}
        mock_timesnet.assert_not_called()

    def test_accepts_tuple_of_quantiles(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["quantiles"] = (0.75, 0.25)
        assert TimesNet_Imputation_Quantile(cfg).quantiles == (0.25, 0.75)


# --------------------------------------------------------------------------- #
# __init__: TimesNet kwargs
# --------------------------------------------------------------------------- #
class TestTimesNetKwargs:
    SHARED_KEYS = (
        "n_steps",
        "n_features",
        "n_layers",
        "top_k",
        "d_model",
        "d_ffn",
        "n_kernels",
        "dropout",
        "apply_nonstationary_norm",
        "batch_size",
        "epochs",
        "patience",
    )

    def test_config_values_forwarded_to_every_model(self, cfg, model, mock_timesnet):
        for call in mock_timesnet.call_args_list:
            for key in self.SHARED_KEYS:
                assert call.kwargs[key] == cfg[key], key

    def test_fixed_arguments(self, model, mock_timesnet):
        for call in mock_timesnet.call_args_list:
            assert call.kwargs["model_saving_strategy"] == "best"
            assert call.kwargs["verbose"] is True

    def test_training_loss_matches_quantile(self, model, mock_timesnet):
        by_q = timesnet_kwargs_by_quantile(mock_timesnet, model.quantiles)
        for q, kwargs in by_q.items():
            assert kwargs["training_loss"].quantile == q

    def test_validation_metric_matches_quantile(self, model, mock_timesnet):
        by_q = timesnet_kwargs_by_quantile(mock_timesnet, model.quantiles)
        for q, kwargs in by_q.items():
            assert kwargs["validation_metric"].quantile == q

    def test_quantile_loss_built_with_keyword_quantile(self, model, mock_qloss):
        # one for training, one for validation, per quantile
        assert mock_qloss.call_count == 2 * len(model.quantiles)
        called_with = sorted(c.kwargs["quantile"] for c in mock_qloss.call_args_list)
        assert called_with == sorted(list(model.quantiles) * 2)

    def test_training_and_validation_losses_are_separate_instances(self, model, mock_timesnet):
        for call in mock_timesnet.call_args_list:
            assert call.kwargs["training_loss"] is not call.kwargs["validation_metric"]

    def test_losses_not_shared_across_models(self, model, mock_timesnet):
        train_losses = [c.kwargs["training_loss"] for c in mock_timesnet.call_args_list]
        assert len({id(x) for x in train_losses}) == len(train_losses)


# --------------------------------------------------------------------------- #
# __init__: optimizer
# --------------------------------------------------------------------------- #
class TestOptimizer:
    def test_adamw_built_from_optimizer_parameters(self, cfg, model, mock_adamw):
        for call in mock_adamw.call_args_list:
            assert call.kwargs == cfg["optimizer_parameters"]

    def test_fresh_optimizer_per_model(self, model, mock_adamw, mock_timesnet):
        assert mock_adamw.call_count == len(model.quantiles)
        optimizers = [c.kwargs["optimizer"] for c in mock_timesnet.call_args_list]
        assert len({id(o) for o in optimizers}) == len(optimizers)

    def test_optimizer_passed_to_timesnet(self, model, mock_adamw, mock_timesnet):
        passed = [c.kwargs["optimizer"] for c in mock_timesnet.call_args_list]
        assert passed == mock_adamw.created  # same objects, same order

    def test_empty_optimizer_parameters_allowed(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["optimizer_parameters"] = {}
        TimesNet_Imputation_Quantile(cfg)
        for call in mock_adamw.call_args_list:
            assert call.kwargs == {}


# --------------------------------------------------------------------------- #
# __init__: saving paths
# --------------------------------------------------------------------------- #
class TestSavingPath:
    def test_none_saving_path(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        m = TimesNet_Imputation_Quantile(cfg)
        assert m.saving_path is None
        for call in mock_timesnet.call_args_list:
            assert call.kwargs["saving_path"] is None

    def test_str_saving_path_converted_to_path(self, cfg, mock_timesnet, mock_adamw, mock_qloss, tmp_path):
        m = TimesNet_Imputation_Quantile(cfg, saving_path=str(tmp_path))
        assert isinstance(m.saving_path, Path)
        assert m.saving_path == tmp_path

    def test_per_quantile_subdirectories(self, cfg, mock_timesnet, mock_adamw, mock_qloss, tmp_path):
        m = TimesNet_Imputation_Quantile(cfg, saving_path=tmp_path)
        by_q = timesnet_kwargs_by_quantile(mock_timesnet, m.quantiles)
        assert by_q[0.1]["saving_path"] == str(tmp_path / "q100")
        assert by_q[0.5]["saving_path"] == str(tmp_path / "q500")
        assert by_q[0.9]["saving_path"] == str(tmp_path / "q900")

    def test_saving_paths_are_strings_and_unique(self, cfg, mock_timesnet, mock_adamw, mock_qloss, tmp_path):
        TimesNet_Imputation_Quantile(cfg, saving_path=tmp_path)
        paths = [c.kwargs["saving_path"] for c in mock_timesnet.call_args_list]
        assert all(isinstance(p, str) for p in paths)
        assert len(set(paths)) == len(paths)

    def test_init_does_not_create_directories(self, cfg, mock_timesnet, mock_adamw, mock_qloss, tmp_path):
        target = tmp_path / "not_created"
        TimesNet_Imputation_Quantile(cfg, saving_path=target)
        assert not target.exists()


# --------------------------------------------------------------------------- #
# __init__: real QuantileLoss (not mocked) validates the quantiles
# --------------------------------------------------------------------------- #
class TestRealQuantileLoss:
    def test_real_losses_carry_quantile(self, cfg, mock_timesnet, mock_adamw):
        m = TimesNet_Imputation_Quantile(cfg)
        by_q = timesnet_kwargs_by_quantile(mock_timesnet, m.quantiles)
        for q, kwargs in by_q.items():
            assert kwargs["training_loss"].quantile == q
            assert kwargs["validation_metric"].quantile == q

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
    def test_out_of_range_quantile_rejected(self, cfg, mock_timesnet, mock_adamw, bad):
        cfg["quantiles"] = [0.5, bad]
        with pytest.raises(AssertionError, match="`quantile` must be in"):
            TimesNet_Imputation_Quantile(cfg)


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
class TestFit:
    def test_every_model_fit_with_same_data(self, model):
        train, val = {"X": np.zeros((2, 3, 4))}, {"X": np.ones((2, 3, 4))}
        model.fit(train, val)
        for m in model.models.values():
            m.fit.assert_called_once_with(train, val)

    def test_val_set_defaults_to_none(self, model):
        train = {"X": np.zeros((2, 3, 4))}
        model.fit(train)
        for m in model.models.values():
            m.fit.assert_called_once_with(train, None)

    def test_accepts_file_paths(self, model):
        model.fit("train.h5", "val.h5")
        for m in model.models.values():
            m.fit.assert_called_once_with("train.h5", "val.h5")

    def test_returns_self(self, model):
        assert model.fit({"X": np.zeros((1, 1, 1))}) is model

    def test_models_fit_in_quantile_order(self, model):
        order = []
        for q, m in model.models.items():
            m.fit.side_effect = lambda *a, _q=q, **k: order.append(_q)
        model.fit({"X": np.zeros((1, 1, 1))})
        assert order == [0.1, 0.5, 0.9]

    def test_exception_in_one_model_propagates_and_stops(self, model):
        model.models[0.5].fit.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            model.fit({"X": np.zeros((1, 1, 1))})
        model.models[0.1].fit.assert_called_once()
        model.models[0.9].fit.assert_not_called()


# --------------------------------------------------------------------------- #
# impute
# --------------------------------------------------------------------------- #
def _set_outputs(model, arrays_by_q):
    for q, arr in arrays_by_q.items():
        model.models[q].impute.return_value = arr


@pytest.fixture
def ordered_outputs():
    """Per-quantile predictions that are already non-crossing: 0.1 < 0.5 < 0.9."""
    base = np.random.default_rng(0).normal(size=(N, T, F))
    return {0.1: base - 1.0, 0.5: base, 0.9: base + 1.0}


class TestImpute:
    def test_output_shape(self, model, ordered_outputs):
        _set_outputs(model, ordered_outputs)
        out = model.impute({"X": np.zeros((N, T, F))})
        assert out.shape == (N, T, F, 3)

    def test_single_quantile_shape(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["quantiles"] = [0.5]
        m = TimesNet_Imputation_Quantile(cfg)
        m.models[0.5].impute.return_value = np.zeros((N, T, F))
        assert m.impute({"X": None}).shape == (N, T, F, 1)

    def test_every_model_called_with_test_set(self, model, ordered_outputs):
        _set_outputs(model, ordered_outputs)
        test_set = {"X": np.zeros((N, T, F))}
        model.impute(test_set)
        for m in model.models.values():
            m.impute.assert_called_once_with(test_set)

    def test_accepts_file_path(self, model, ordered_outputs):
        _set_outputs(model, ordered_outputs)
        model.impute("test.h5")
        for m in model.models.values():
            m.impute.assert_called_once_with("test.h5")

    def test_quantile_axis_order_follows_sorted_quantiles(self, model, ordered_outputs):
        _set_outputs(model, ordered_outputs)
        out = model.impute({"X": None}, sort=False)
        np.testing.assert_array_equal(out[..., 0], ordered_outputs[0.1])
        np.testing.assert_array_equal(out[..., 1], ordered_outputs[0.5])
        np.testing.assert_array_equal(out[..., 2], ordered_outputs[0.9])

    def test_sort_defaults_to_true(self, model):
        # make the 0.1 model predict *higher* than the 0.9 model everywhere
        _set_outputs(model, {0.1: np.full((N, T, F), 3.0), 0.5: np.full((N, T, F), 2.0), 0.9: np.full((N, T, F), 1.0)})
        out = model.impute({"X": None})
        assert np.all(np.diff(out, axis=-1) >= 0)

    def test_sort_true_fixes_crossing_quantiles(self, model):
        _set_outputs(model, {0.1: np.full((N, T, F), 3.0), 0.5: np.full((N, T, F), 2.0), 0.9: np.full((N, T, F), 1.0)})
        out = model.impute({"X": None}, sort=True)
        np.testing.assert_array_equal(out[0, 0, 0], [1.0, 2.0, 3.0])

    def test_sort_false_preserves_crossing(self, model):
        _set_outputs(model, {0.1: np.full((N, T, F), 3.0), 0.5: np.full((N, T, F), 2.0), 0.9: np.full((N, T, F), 1.0)})
        out = model.impute({"X": None}, sort=False)
        np.testing.assert_array_equal(out[0, 0, 0], [3.0, 2.0, 1.0])

    def test_sort_is_noop_when_already_ordered(self, model, ordered_outputs):
        _set_outputs(model, ordered_outputs)
        a = model.impute({"X": None}, sort=True)
        b = model.impute({"X": None}, sort=False)
        np.testing.assert_array_equal(a, b)

    def test_sort_only_along_quantile_axis(self, model):
        # crossing happens only at one (sample, step, feature) location
        lo = np.zeros((N, T, F))
        mid = np.ones((N, T, F))
        hi = np.full((N, T, F), 2.0)
        lo[1, 2, 3], hi[1, 2, 3] = 5.0, -5.0
        _set_outputs(model, {0.1: lo, 0.5: mid, 0.9: hi})
        out = model.impute({"X": None}, sort=True)
        np.testing.assert_array_equal(out[1, 2, 3], [-5.0, 1.0, 5.0])
        np.testing.assert_array_equal(out[0, 0, 0], [0.0, 1.0, 2.0])

    def test_sorted_output_preserves_values(self, model):
        rng = np.random.default_rng(5)
        outs = {q: rng.normal(size=(N, T, F)) for q in model.quantiles}
        _set_outputs(model, outs)
        stacked = np.stack([outs[q] for q in model.quantiles], axis=-1)
        result = model.impute({"X": None}, sort=True)
        np.testing.assert_allclose(result, np.sort(stacked, axis=-1))

    def test_empty_quantiles_raises(self, cfg, mock_timesnet, mock_adamw, mock_qloss):
        cfg["quantiles"] = []
        m = TimesNet_Imputation_Quantile(cfg)
        with pytest.raises(ValueError):
            m.impute({"X": None})

    def test_exception_propagates(self, model):
        model.models[0.5].impute.side_effect = ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            model.impute({"X": None})


# --------------------------------------------------------------------------- #
# save
# --------------------------------------------------------------------------- #
class TestSave:
    def test_each_model_saved_to_tagged_file(self, model, tmp_path):
        model.save(tmp_path)
        model.models[0.1].save.assert_called_once_with(str(tmp_path / "q100.pypots"), overwrite=False)
        model.models[0.5].save.assert_called_once_with(str(tmp_path / "q500.pypots"), overwrite=False)
        model.models[0.9].save.assert_called_once_with(str(tmp_path / "q900.pypots"), overwrite=False)

    def test_overwrite_forwarded_to_all(self, model, tmp_path):
        model.save(tmp_path, overwrite=True)
        for m in model.models.values():
            assert m.save.call_args.kwargs["overwrite"] is True

    def test_creates_directory(self, model, tmp_path):
        target = tmp_path / "missing"
        model.save(target)
        assert target.is_dir()

    def test_creates_nested_directories(self, model, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        model.save(target)
        assert target.is_dir()

    def test_existing_directory_ok(self, model, tmp_path):
        model.save(tmp_path)
        model.save(tmp_path)  # must not raise
        for m in model.models.values():
            assert m.save.call_count == 2

    def test_accepts_str_directory(self, model, tmp_path):
        model.save(str(tmp_path))
        model.models[0.5].save.assert_called_once_with(str(tmp_path / "q500.pypots"), overwrite=False)

    def test_paths_passed_to_models_are_strings_and_unique(self, model, tmp_path):
        model.save(tmp_path)
        paths = [m.save.call_args.args[0] for m in model.models.values()]
        assert all(isinstance(p, str) for p in paths)
        assert len(set(paths)) == len(paths)

    def test_exception_propagates(self, model, tmp_path):
        model.models[0.5].save.side_effect = FileExistsError("exists")
        with pytest.raises(FileExistsError):
            model.save(tmp_path)


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
class TestLoad:
    def test_each_model_loaded_from_tagged_file(self, model, tmp_path):
        model.load(tmp_path)
        model.models[0.1].load.assert_called_once_with(str(tmp_path / "q100.pypots"))
        model.models[0.5].load.assert_called_once_with(str(tmp_path / "q500.pypots"))
        model.models[0.9].load.assert_called_once_with(str(tmp_path / "q900.pypots"))

    def test_accepts_str_directory(self, model, tmp_path):
        model.load(str(tmp_path))
        model.models[0.9].load.assert_called_once_with(str(tmp_path / "q900.pypots"))

    def test_does_not_create_directory(self, model, tmp_path):
        target = tmp_path / "missing"
        model.load(target)
        assert not target.exists()

    def test_returns_none(self, model, tmp_path):
        assert model.load(tmp_path) is None

    def test_exception_propagates(self, model, tmp_path):
        model.models[0.1].load.side_effect = FileNotFoundError("no such file")
        with pytest.raises(FileNotFoundError):
            model.load(tmp_path)


# --------------------------------------------------------------------------- #
# save / load consistency
# --------------------------------------------------------------------------- #
def test_save_and_load_use_same_filenames_per_quantile(model, tmp_path):
    model.save(tmp_path)
    model.load(tmp_path)
    for m in model.models.values():
        assert m.save.call_args.args[0] == m.load.call_args.args[0]


def test_saving_path_subdir_names_match_checkpoint_stems(cfg, mock_timesnet, mock_adamw, mock_qloss, tmp_path):
    """Per-model `saving_path` dirs and explicit save files share the same tag."""
    m = TimesNet_Imputation_Quantile(cfg, saving_path=tmp_path)
    m.save(tmp_path)
    for (q, inner), call in zip(m.models.items(), mock_timesnet.call_args_list):
        tag = TimesNet_Imputation_Quantile._tag(q)
        assert call.kwargs["saving_path"].endswith(tag)
        assert Path(inner.save.call_args.args[0]).stem == tag