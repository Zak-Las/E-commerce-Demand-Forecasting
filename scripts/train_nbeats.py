"""Minimal N-BEATS training script (placeholder dataset).

Replace synthetic data loader with real feature engineered dataset once ready.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from src.models.nbeats_module import NBeatsModule, NBeatsConfig


def get_synthetic_dataloader(cfg: NBeatsConfig, batch_size: int = 32, n_batches: int = 128):
    X = torch.randn(n_batches * batch_size, cfg.input_length)
    Y = torch.randn(n_batches * batch_size, cfg.forecast_length)
    ds = TensorDataset(X, Y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def main():
    cfg = NBeatsConfig()
    model = NBeatsModule(cfg)
    train_loader = get_synthetic_dataloader(cfg)
    val_loader = get_synthetic_dataloader(cfg, n_batches=8)

    trainer = pl.Trainer(max_epochs=2, accelerator="auto", devices=1, deterministic=True, log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)

    out_path = "models/nbeats_synth.pt"
    model.cpu()
    model.eval()
    out_path_parent = __import__('pathlib').Path(out_path).parent
    out_path_parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
