
import torch
import numpy as np
from uncertainty_aware_planner.models.forecasting.dlinear_quantile import (
    DLinearQuantile,
    pinball_loss,
    fit, 
    predict_quantiles
)
from pathlib import Path
import pandas as pd


def test_dlinear_quantile_output():
    model = DLinearQuantile(
        encoder_days=21,
        horizon_days=7,
        quantiles=(0.025, 0.5, 0.975),
    )

    x = torch.rand(4, 21)
    predictions = model(x)

    assert predictions.shape == (4, 7, 3)
    assert torch.isfinite(predictions).all()

    targets = torch.rand(4, 7)
    loss = pinball_loss(
        predictions, targets, model.quantiles
    )

    assert torch.isfinite(loss)

    loss.backward()
    assert model.seasonal_linear.weight.grad is not None


def test_final_encoder_length():
    model = DLinearQuantile(encoder_days=70)
    predictions = model(torch.rand(4, 70))

    assert predictions.shape == (4, 7, 3)


def test_fit_saves_checkpoint(tmp_path):
    class FakeDataset:
        def to_dataloader(self, **kwargs):
            x = {"encoder_target": torch.rand(4, 21)}
            y = (torch.rand(4, 7), None)
            return [(x, y)]

    model = DLinearQuantile(encoder_days=21)

    cfg = {
        "dataloader": {
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        },
        "trainer": {"max_epochs": 1},
    }

    checkpoint = fit(
        model,
        FakeDataset(),
        cfg,
        tmp_path,
    )

    assert Path(checkpoint).exists()
    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    assert "model_state_dict" in saved



def test_predict_quantiles(tmp_path):
    class FakeDataset:
        def to_dataloader(self, train, batch_size, num_workers):
            x = {
                "encoder_target": torch.rand(4, 21),
                "decoder_lengths": torch.full((4,), 7),
            }
            return [(x, None)]

        def x_to_index(self, x):
            return pd.DataFrame({
                "store_id": [1, 1, 2, 2],
                "product_id": [10, 11, 10, 11],
                "time_idx": [28, 28, 28, 28],
            })

    model = DLinearQuantile(encoder_days=21)

    checkpoint_path = tmp_path / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "encoder_days": 21,
        "horizon_days": 7,
        "quantiles": model.quantiles,
        "moving_avg": model.moving_avg,
    }, checkpoint_path)

    cfg = {
        "dataloader": {
            "batch_size": 4,
            "num_workers": 0,
        }
    }

    q, index, lengths = predict_quantiles(
        checkpoint_path,
        FakeDataset(),
        cfg,
    )

    assert q.shape == (4, 7, 3)
    assert len(index) == 4
    assert lengths.tolist() == [7, 7, 7, 7]
    assert np.isfinite(q).all()
