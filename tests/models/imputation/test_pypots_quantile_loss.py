"""Tests for the custom masked quantile loss.

Adjust the import below to match the module name of the file under test.
Generated with Claude
"""
import numpy as np
import pytest
import torch

from uncertainty_aware_planner.models.imputation.pypots_quantile_loss import QuantileLoss, _check_inputs, calc_quantile_loss


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _np(x, dtype=np.float64):
    return np.array(x, dtype=dtype)


def _t(x, dtype=torch.float64, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)


@pytest.fixture
def simple_case():
    """err = targets - predictions = [1, 0, -2]."""
    preds = [1.0, 2.0, 3.0]
    targets = [2.0, 2.0, 1.0]
    return preds, targets


# --------------------------------------------------------------------------- #
# _check_inputs
# --------------------------------------------------------------------------- #
class TestCheckInputs:
    def test_returns_numpy_for_numpy_inputs(self):
        lib = _check_inputs(_np([1, 2]), _np([1, 2]))
        assert lib is np

    def test_returns_torch_for_tensor_inputs(self):
        lib = _check_inputs(_t([1, 2]), _t([1, 2]))
        assert lib is torch

    def test_type_mismatch_raises(self):
        with pytest.raises(AssertionError, match="types of `predictions` and `targets`"):
            _check_inputs(_np([1, 2]), _t([1, 2]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError, match="shape of `predictions` and `targets`"):
            _check_inputs(_np([1, 2, 3]), _np([1, 2]))

    def test_shape_mismatch_allowed_when_check_shape_false(self):
        lib = _check_inputs(_np([1, 2, 3]), _np([1, 2]), check_shape=False)
        assert lib is np

    @pytest.mark.parametrize("make", [_np, _t])
    def test_nan_in_predictions_raises(self, make):
        with pytest.raises(AssertionError, match="`predictions` mustn't contain NaN"):
            _check_inputs(make([1.0, float("nan")]), make([1.0, 2.0]))

    @pytest.mark.parametrize("make", [_np, _t])
    def test_nan_in_targets_raises(self, make):
        with pytest.raises(AssertionError, match="`targets` mustn't contain NaN"):
            _check_inputs(make([1.0, 2.0]), make([1.0, float("nan")]))

    @pytest.mark.parametrize("make", [_np, _t])
    def test_nan_in_masks_raises(self, make):
        with pytest.raises(AssertionError, match="`masks` mustn't contain NaN"):
            _check_inputs(make([1.0, 2.0]), make([1.0, 2.0]), make([1.0, float("nan")]))

    def test_mask_type_mismatch_raises(self):
        with pytest.raises(AssertionError, match="types of `masks`"):
            _check_inputs(_np([1, 2]), _np([1, 2]), _t([1, 1]))

    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(AssertionError, match="shape of `masks`"):
            _check_inputs(_np([1, 2]), _np([1, 2]), _np([1, 1, 1]))

    def test_valid_mask_passes(self):
        assert _check_inputs(_t([1, 2]), _t([1, 2]), _t([1, 0])) is torch

    def test_integer_mask_passes(self):
        mask = np.array([1, 0], dtype=np.int64)
        assert _check_inputs(_np([1, 2]), _np([1, 2]), mask) is np


# --------------------------------------------------------------------------- #
# calc_quantile_loss: input validation
# --------------------------------------------------------------------------- #
class TestCalcQuantileLossValidation:
    @pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_quantile_raises(self, q):
        with pytest.raises(AssertionError, match="`quantile` must be in"):
            calc_quantile_loss(_np([1.0]), _np([1.0]), q)

    def test_mixed_types_raise(self):
        with pytest.raises(AssertionError):
            calc_quantile_loss(_np([1.0]), _t([1.0]), 0.5)

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            calc_quantile_loss(_np([1.0, 2.0]), _np([1.0]), 0.5)

    def test_nan_input_raises(self):
        with pytest.raises(AssertionError):
            calc_quantile_loss(_t([float("nan")]), _t([1.0]), 0.5)

    def test_mask_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            calc_quantile_loss(_np([1.0, 2.0]), _np([1.0, 2.0]), 0.5, _np([1.0]))


# --------------------------------------------------------------------------- #
# calc_quantile_loss: numerical correctness
# --------------------------------------------------------------------------- #
class TestCalcQuantileLossValues:
    @pytest.mark.parametrize("make", [_np, _t])
    def test_hand_computed_unmasked(self, make, simple_case):
        preds, targets = simple_case
        # q = 0.9, err = [1, 0, -2] -> loss = [0.9, 0, 0.2]
        out = calc_quantile_loss(make(preds), make(targets), 0.9)
        assert float(out) == pytest.approx((0.9 + 0.0 + 0.2) / 3)

    @pytest.mark.parametrize("make", [_np, _t])
    def test_hand_computed_masked(self, make, simple_case):
        preds, targets = simple_case
        masks = make([1.0, 0.0, 1.0])
        out = calc_quantile_loss(make(preds), make(targets), 0.9, masks)
        assert float(out) == pytest.approx((0.9 + 0.2) / 2)

    @pytest.mark.parametrize("make", [_np, _t])
    def test_perfect_prediction_is_zero(self, make):
        x = make([0.5, -1.0, 3.0])
        assert float(calc_quantile_loss(x, x, 0.3)) == pytest.approx(0.0)

    def test_median_equals_half_mae(self):
        rng = np.random.default_rng(0)
        p, t = rng.normal(size=(4, 5)), rng.normal(size=(4, 5))
        out = calc_quantile_loss(p, t, 0.5)
        assert out == pytest.approx(0.5 * np.mean(np.abs(t - p)))

    def test_under_prediction_weighted_by_q(self):
        # prediction below target -> err > 0 -> loss = q * err
        out = calc_quantile_loss(_np([0.0]), _np([2.0]), 0.8)
        assert out == pytest.approx(0.8 * 2.0)

    def test_over_prediction_weighted_by_one_minus_q(self):
        # prediction above target -> err < 0 -> loss = (1 - q) * |err|
        out = calc_quantile_loss(_np([2.0]), _np([0.0]), 0.8)
        assert out == pytest.approx(0.2 * 2.0)

    def test_loss_is_non_negative(self):
        rng = np.random.default_rng(1)
        p, t = rng.normal(size=100), rng.normal(size=100)
        for q in (0.05, 0.5, 0.95):
            assert calc_quantile_loss(p, t, q) >= 0.0

    def test_symmetry_between_q_and_one_minus_q(self):
        rng = np.random.default_rng(2)
        p, t = rng.normal(size=50), rng.normal(size=50)
        assert calc_quantile_loss(p, t, 0.2) == pytest.approx(calc_quantile_loss(t, p, 0.8))

    def test_numpy_and_torch_agree(self):
        rng = np.random.default_rng(3)
        p, t = rng.normal(size=(3, 4, 2)), rng.normal(size=(3, 4, 2))
        m = (rng.random(size=(3, 4, 2)) > 0.4).astype(np.float64)
        np_out = calc_quantile_loss(p, t, 0.7, m)
        torch_out = calc_quantile_loss(torch.from_numpy(p), torch.from_numpy(t), 0.7, torch.from_numpy(m))
        assert float(torch_out) == pytest.approx(float(np_out))

    @pytest.mark.parametrize("make", [_np, _t])
    def test_all_ones_mask_equals_unmasked(self, make, simple_case):
        preds, targets = simple_case
        unmasked = calc_quantile_loss(make(preds), make(targets), 0.25)
        masked = calc_quantile_loss(make(preds), make(targets), 0.25, make([1.0, 1.0, 1.0]))
        assert float(masked) == pytest.approx(float(unmasked))

    @pytest.mark.parametrize("make", [_np, _t])
    def test_all_zero_mask_returns_zero_not_nan(self, make, simple_case):
        preds, targets = simple_case
        out = float(calc_quantile_loss(make(preds), make(targets), 0.9, make([0.0, 0.0, 0.0])))
        assert out == pytest.approx(0.0)
        assert not np.isnan(out)

    def test_masked_positions_do_not_affect_loss(self):
        p = _np([1.0, 2.0, 3.0])
        m = _np([1.0, 1.0, 0.0])
        t1 = _np([2.0, 2.0, 100.0])
        t2 = _np([2.0, 2.0, -100.0])
        assert calc_quantile_loss(p, t1, 0.6, m) == pytest.approx(calc_quantile_loss(p, t2, 0.6, m))

    def test_multidimensional_inputs(self):
        p = np.zeros((2, 3, 4))
        t = np.ones((2, 3, 4))
        # err = 1 everywhere -> loss = q
        assert calc_quantile_loss(p, t, 0.35) == pytest.approx(0.35)

    def test_integer_mask_works(self, simple_case):
        preds, targets = simple_case
        masks = np.array([1, 0, 1], dtype=np.int64)
        out = calc_quantile_loss(_np(preds), _np(targets), 0.9, masks)
        assert out == pytest.approx((0.9 + 0.2) / 2)


# --------------------------------------------------------------------------- #
# calc_quantile_loss: output types / autograd
# --------------------------------------------------------------------------- #
class TestCalcQuantileLossOutputs:
    def test_numpy_returns_scalar(self, simple_case):
        preds, targets = simple_case
        out = calc_quantile_loss(_np(preds), _np(targets), 0.5)
        assert np.ndim(out) == 0

    def test_torch_returns_zero_dim_tensor(self, simple_case):
        preds, targets = simple_case
        out = calc_quantile_loss(_t(preds), _t(targets), 0.5)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_torch_dtype_preserved(self):
        p = torch.zeros(3, dtype=torch.float32)
        t = torch.ones(3, dtype=torch.float32)
        assert calc_quantile_loss(p, t, 0.5).dtype == torch.float32

    def test_gradient_flows_to_predictions(self):
        # err = [1, -1] -> d loss / d pred = [-q, (1 - q)] / N
        q = 0.9
        p = _t([0.0, 2.0], requires_grad=True)
        t = _t([1.0, 1.0])
        calc_quantile_loss(p, t, q).backward()
        expected = torch.tensor([-q, 1 - q], dtype=torch.float64) / 2
        assert torch.allclose(p.grad, expected)

    def test_gradient_zero_at_masked_positions(self):
        p = _t([0.0, 2.0, 5.0], requires_grad=True)
        t = _t([1.0, 1.0, 1.0])
        m = _t([1.0, 1.0, 0.0])
        calc_quantile_loss(p, t, 0.9, m).backward()
        assert p.grad[2].item() == 0.0
        assert p.grad[:2].abs().sum().item() > 0.0


# --------------------------------------------------------------------------- #
# QuantileLoss module
# --------------------------------------------------------------------------- #
class TestQuantileLossModule:
    def test_default_quantile_is_median(self):
        assert QuantileLoss().quantile == 0.5

    def test_custom_quantile_stored(self):
        assert QuantileLoss(0.9).quantile == 0.9

    @pytest.mark.parametrize("q", [0.0, 1.0, -1.0, 2.0])
    def test_invalid_quantile_raises(self, q):
        with pytest.raises(AssertionError, match="`quantile` must be in"):
            QuantileLoss(q)

    def test_is_torch_module(self):
        assert isinstance(QuantileLoss(), torch.nn.Module)

    def test_forward_matches_function_unmasked(self, simple_case):
        preds, targets = simple_case
        p, t = _t(preds), _t(targets)
        crit = QuantileLoss(0.9)
        assert torch.allclose(crit(p, t), calc_quantile_loss(p, t, 0.9))

    def test_forward_matches_function_masked(self, simple_case):
        preds, targets = simple_case
        p, t, m = _t(preds), _t(targets), _t([1.0, 0.0, 1.0])
        crit = QuantileLoss(0.9)
        assert torch.allclose(crit(p, t, m), calc_quantile_loss(p, t, 0.9, m))

    def test_forward_hand_computed(self, simple_case):
        preds, targets = simple_case
        out = QuantileLoss(0.9)(_t(preds), _t(targets))
        assert out.item() == pytest.approx(1.1 / 3)

    def test_forward_accepts_masks_keyword(self, simple_case):
        preds, targets = simple_case
        out = QuantileLoss(0.9)(_t(preds), _t(targets), masks=_t([1.0, 0.0, 1.0]))
        assert out.item() == pytest.approx(0.55)

    def test_backward_through_module(self):
        p = _t([0.0, 2.0], requires_grad=True)
        t = _t([1.0, 1.0])
        QuantileLoss(0.75)(p, t).backward()
        assert p.grad is not None
        assert torch.allclose(p.grad, torch.tensor([-0.75, 0.25], dtype=torch.float64) / 2)

    def test_forward_rejects_numpy_inputs_with_mismatched_types(self):
        with pytest.raises(AssertionError):
            QuantileLoss(0.5)(_t([1.0]), _np([1.0]))