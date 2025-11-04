"""Minimal N-BEATS style LightningModule.

Implements simplified residual stacking with blocks producing
backcast (for residual subtraction) and forecast (target horizon).

Simplifications vs original paper:
    - Generic fully-connected block (no explicit trend/seasonality bases yet)
    - Identity basis: theta_backcast directly sized to input_length;
        theta_forecast directly sized to forecast_length.
    - No weight sharing between stacks.
    - Loss: MSE; logs MAE and WAPE for monitoring.

Roadmap extensions:
    - Add basis functions (polynomial trend, Fourier seasonality)
    - Quantile heads for probabilistic forecasts
    - Per-item embedding inputs
    - Mixed horizon support
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
import pytorch_lightning as pl


@dataclass
class NBeatsConfig:
    input_length: int = 28 * 4  # lookback window length
    forecast_length: int = 30   # horizon
    hidden_dim: int = 256
    num_stacks: int = 4
    num_blocks_per_stack: int = 3
    n_layers: int = 4           # FC layers per block
    layer_width: int = 512
    learning_rate: float = 1e-3
    dropout: float = 0.0


class NBeatsBlock(nn.Module):
    """Generic fully-connected block outputting backcast & forecast."""

    def __init__(self, input_length: int, forecast_length: int, layer_width: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_f = input_length
        for i in range(n_layers):
            layers.append(nn.Linear(in_f if i == 0 else layer_width, layer_width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(layer_width, input_length)
        self.forecast_head = nn.Linear(layer_width, forecast_length)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class NBeatsModule(pl.LightningModule):
    def __init__(self, config: Optional[NBeatsConfig] = None):
        super().__init__()
        self.config = config or NBeatsConfig()
        # Log hyperparameters (Lightning will recurse object repr)
        self.save_hyperparameters({"input_length": self.config.input_length,
                                   "forecast_length": self.config.forecast_length,
                                   "num_stacks": self.config.num_stacks,
                                   "num_blocks_per_stack": self.config.num_blocks_per_stack,
                                   "n_layers": self.config.n_layers,
                                   "layer_width": self.config.layer_width,
                                   "learning_rate": self.config.learning_rate,
                                   "dropout": self.config.dropout})
        # Build stacks
        self.stacks = nn.ModuleList([
            nn.ModuleList([
                NBeatsBlock(
                    input_length=self.config.input_length,
                    forecast_length=self.config.forecast_length,
                    layer_width=self.config.layer_width,
                    n_layers=self.config.n_layers,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_blocks_per_stack)
            ])
            for _ in range(self.config.num_stacks)
        ])
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual stacking: subtract backcast progressively
        residual = x
        forecast_total = torch.zeros(x.size(0), self.config.forecast_length, device=x.device)
        for stack in self.stacks:
            for block in stack:
                backcast, forecast = block(residual)
                residual = residual - backcast
                forecast_total = forecast_total + forecast
        return forecast_total

    def _metrics(self, y_hat: torch.Tensor, y: torch.Tensor) -> dict:
        mae = torch.mean(torch.abs(y_hat - y))
        denom = torch.sum(torch.abs(y))
        wape = torch.nan if denom == 0 else 100.0 * torch.sum(torch.abs(y_hat - y)) / denom
        return {"mae": mae, "wape": wape}

    def training_step(self, batch, batch_idx: int):  # type: ignore
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        m = self._metrics(y_hat, y)
        # Log only on epoch to avoid noisy per-step spam and enable proper callback aggregation
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", m["mae"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_wape", m["wape"], prog_bar=True, on_step=False, on_epoch=True)
        # Derive accuracy as (100 - WAPE) when WAPE is not NaN
        if not torch.isnan(m["wape"]):
            train_accuracy = 100.0 - m["wape"]
            self.log("train_accuracy", train_accuracy, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        m = self._metrics(y_hat, y)
        # Log epoch-level metrics (EarlyStopping / Checkpoint monitor 'val_loss')
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", m["mae"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_wape", m["wape"], prog_bar=True, on_step=False, on_epoch=True)
        if not torch.isnan(m["wape"]):
            val_accuracy = 100.0 - m["wape"]
            self.log("val_accuracy", val_accuracy, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    @staticmethod
    def example_batch(batch_size: int = 4, config: Optional[NBeatsConfig] = None):
        cfg = config or NBeatsConfig()
        x = torch.randn(batch_size, cfg.input_length)
        y = torch.randn(batch_size, cfg.forecast_length)
        return x, y


if __name__ == "__main__":  # simple smoke
    cfg = NBeatsConfig()
    model = NBeatsModule(cfg)
    x, y = model.example_batch()
    out = model(x)
    print("Output shape", out.shape)
