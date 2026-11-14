
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd


class DLinearQuantile(nn.Module):
    """DLinear forecasting model with multiple quantile outputs."""

    def __init__(
        self,
        encoder_days: int,
        horizon_days: int = 7,
        quantiles=(0.025, 0.5, 0.975),
        moving_avg: int = 7,
    ):
        super().__init__()

        if moving_avg < 1 or moving_avg % 2 == 0:
            raise ValueError("moving_avg must be a positive odd number")

        self.encoder_days = encoder_days
        self.horizon_days = horizon_days
        self.quantiles = tuple(quantiles)
        self.moving_avg = moving_avg

        output_size = horizon_days * len(quantiles)

        self.seasonal_linear = nn.Linear(encoder_days, output_size)
        self.trend_linear = nn.Linear(encoder_days, output_size)

    def forward(self, x):
        # x: [batch, encoder_days]
        if x.ndim != 2 or x.shape[1] != self.encoder_days:
            raise ValueError("Unexpected input shape")

        # Moving-average trend
        series = x.unsqueeze(1)
        padding = self.moving_avg // 2

        padded = F.pad(series, (padding, padding), mode="replicate")

        trend = F.avg_pool1d(
            padded,
            kernel_size=self.moving_avg,
            stride=1,
        ).squeeze(1)

        # Seasonal component
        seasonal = x - trend

        # Predict future demand quantiles
        output = (
            self.seasonal_linear(seasonal)
            + self.trend_linear(trend)
        )

        return output.reshape(
            -1, self.horizon_days, len(self.quantiles)
        )


def pinball_loss(predictions, targets, quantiles):
    """Quantile regression loss, using [batch, horizon, quantiles]."""

    q = predictions.new_tensor(quantiles).view(1, 1, -1)
    errors = targets.unsqueeze(-1) - predictions

    return torch.maximum(
        q * errors,
        (q - 1) * errors,
    ).mean()


def build_model(train_ds, cfg):
    """Build DLinear Quantile using the team's dataset and config."""

    encoder_days = train_ds.max_encoder_length
    horizon_days = train_ds.max_prediction_length

    return DLinearQuantile(
        encoder_days=encoder_days,
        horizon_days=horizon_days,
        quantiles=cfg["quantiles"],
        moving_avg=cfg["model"]["moving_avg"],
    )


def fit(model, train_ds, cfg, run_dir):
    """Train DLinear Quantile and save its checkpoint."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    loader_cfg = cfg["dataloader"]

    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size=loader_cfg["batch_size"],
        num_workers=loader_cfg["num_workers"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["model"]["learning_rate"],
        weight_decay=cfg["model"].get("weight_decay", 0.0),
    )

    epochs = cfg["trainer"]["max_epochs"]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        limit = cfg["trainer"].get("limit_train_batches")

        for batch_idx, (x, y) in enumerate(train_loader):
            if limit is not None and batch_idx >= limit:
                break
            
            history = x["encoder_target"].float().to(device)
            target = y[0].float().to(device)

            optimizer.zero_grad()

            predictions = model(history)

            loss = pinball_loss(
                predictions,
                target,
                model.quantiles,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Pinball Loss: {avg_loss:.4f}"
        )

    checkpoint_path = run_dir / "dlinear_quantile.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_days": model.encoder_days,
            "horizon_days": model.horizon_days,
            "quantiles": model.quantiles,
            "moving_avg": model.moving_avg,
        },
        checkpoint_path,
    )

    return checkpoint_path



def predict_quantiles(
    ckpt_path: Path,
    eval_ds,
    cfg: dict,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load DLinear and predict quantiles for evaluation data."""

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=True,
    )

    model = DLinearQuantile(
        encoder_days=checkpoint["encoder_days"],
        horizon_days=checkpoint["horizon_days"],
        quantiles=checkpoint["quantiles"],
        moving_avg=checkpoint["moving_avg"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = eval_ds.to_dataloader(
        train=False,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
    )

    predictions = []
    indices = []
    lengths = []

    limit = cfg.get("trainer", {}).get("limit_predict_batches")

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):

            # Limit prediction batches during smoke tests
            if limit is not None and batch_idx >= limit:
                break

            # Print progress
            if batch_idx % 100 == 0:
                print(
                    f"Predicting batch {batch_idx}...",
                    flush=True,
                )

            # Historical demand input
            history = x["encoder_target"].float()

            # Predict quantiles
            q = model(history)

            predictions.append(q.cpu().numpy())

            # Recover store/product identifiers
            indices.append(eval_ds.x_to_index(x))

            # Number of valid forecast days
            lengths.append(
                x["decoder_lengths"].cpu().numpy()
            )

    if not predictions:
        raise ValueError("Evaluation dataset is empty")

    return (
        np.concatenate(predictions, axis=0),
        pd.concat(indices, ignore_index=True),
        np.concatenate(lengths, axis=0),
    )
