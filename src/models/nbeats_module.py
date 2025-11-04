"""N-BEATS LightningModule placeholder.

Implements skeleton to be filled with actual block definitions.
Focus: Clean interface for training & inference.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import pytorch_lightning as pl


@dataclass
class NBeatsConfig:
    input_length: int = 28 * 4  # lookback window
    forecast_length: int = 30
    hidden_dim: int = 256
    num_stacks: int = 4
    num_blocks_per_stack: int = 3
    learning_rate: float = 1e-3


class SimpleBlock(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, forecast_length: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.theta = nn.Linear(hidden_dim, forecast_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return self.theta(h)


class NBeatsModule(pl.LightningModule):
    def __init__(self, config: NBeatsConfig | None = None):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])  # lightning logging
        self.config = config or NBeatsConfig()
        in_features = self.config.input_length
        self.blocks = nn.ModuleList([
            SimpleBlock(in_features, self.config.hidden_dim, self.config.forecast_length)
            for _ in range(self.config.num_stacks * self.config.num_blocks_per_stack)
        ])
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_length)
        preds = [b(x) for b in self.blocks]
        # naive aggregation: average outputs (placeholder for residual approach)
        return torch.stack(preds, dim=0).mean(dim=0)

    def training_step(self, batch, batch_idx: int):  # type: ignore
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):  # type: ignore
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

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
