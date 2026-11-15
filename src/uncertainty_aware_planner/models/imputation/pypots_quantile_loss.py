# custom masked quantile loss that can be used with PyPOTS
from typing import Union, Optional
import torch
import numpy as np
from numpy.typing import NDArray
from pypots.nn.modules.loss import Criterion
# from pypots.nn.functional.error import _check_inputs


# directly from PyPOTS
def _check_inputs(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    check_shape: bool = True,
):
    # check type
    assert isinstance(predictions, type(targets)), (
        f"types of `predictions` and `targets` must match, but got"
        f"`predictions`: {type(predictions)}, `target`: {type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    # check shape
    prediction_shape = predictions.shape
    target_shape = targets.shape
    if check_shape:
        assert prediction_shape == target_shape, (
            f"shape of `predictions` and `targets` must match, but got {prediction_shape} and {target_shape}"
        )
    # check NaN
    assert not lib.isnan(predictions).any(), (
        "`predictions` mustn't contain NaN values, but detected NaN in it"
    )
    assert not lib.isnan(targets).any(), (
        "`targets` mustn't contain NaN values, but detected NaN in it"
    )

    if masks is not None:
        # check type
        assert isinstance(masks, type(targets)), (
            f"types of `masks`, `predictions`, and `targets` must match, but got"
            f"`masks`: {type(masks)}, `targets`: {type(targets)}"
        )
        # check shape, masks shape must match targets
        mask_shape = masks.shape
        assert mask_shape == target_shape, (
            f"shape of `masks` must match `targets` shape, "
            f"but got `mask`: {mask_shape} that is different from `targets`: {target_shape}"
        )
        # check NaN
        assert not lib.isnan(masks).any(), (
            "`masks` mustn't contain NaN values, but detected NaN in it"
        )

    return lib


def calc_quantile_loss(
    predictions: Union[NDArray, torch.Tensor],
    targets: Union[NDArray, torch.Tensor],
    quantile: float,
    masks: Optional[Union[NDArray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Masked quantile (pinball) loss between ``predictions`` and ``targets``.

    Parameters
    ----------
    predictions : predicted values for the given quantile.
    targets : ground-truth values.
    quantile : float in (0, 1), exclusive.
    masks : optional; only positions where masks == 1 contribute.
    """
    assert 0.0 < quantile < 1.0, f"`quantile` must be in (0, 1), got {quantile}"

    lib = _check_inputs(predictions, targets, masks)

    err = targets - predictions
    # pinball loss: q * err if err >= 0 else (q - 1) * err
    loss = lib.maximum(quantile * err, (quantile - 1) * err)

    if masks is not None:
        return lib.sum(loss * masks) / (lib.sum(masks) + 1e-12)
    return lib.mean(loss)


class QuantileLoss(Criterion):
    def __init__(self, quantile: float = 0.5):
        super().__init__()
        assert 0.0 < quantile < 1.0, f"`quantile` must be in (0, 1), got {quantile}"
        self.quantile = quantile

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return calc_quantile_loss(logits, targets, self.quantile, masks)
