"""Minimal N-BEATS training script (placeholder dataset).

Replace synthetic data loader with real feature engineered dataset once ready.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from src.models.nbeats_module import NBeatsModule, NBeatsConfig
import numpy as np
from pathlib import Path
import argparse


def load_feature_npz(path: Path, cfg: NBeatsConfig):
    if not path.exists():
        raise FileNotFoundError(f"Feature file {path} not found. Run build_features.py first.")
    data = np.load(path)
    X = data["X"]  # shape (N, n_features * input_length)
    Y = data["Y"]  # shape (N, forecast_length)
    # If mismatch in forecast length, truncate or raise
    if Y.shape[1] != cfg.forecast_length:
        raise ValueError(
            f"Forecast length mismatch: features have {Y.shape[1]}, cfg expects {cfg.forecast_length}"
        )
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def build_dataloaders(feature_path: Path, cfg: NBeatsConfig, batch_size: int = 64, val_frac: float = 0.05):
    X, Y = load_feature_npz(feature_path, cfg)
    ds = TensorDataset(X, Y)
    val_size = int(len(ds) * val_frac)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


def parse_args():
    ap = argparse.ArgumentParser(description="Train N-BEATS model using precomputed features")
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", type=Path, default=Path("models/nbeats_v1.pt"))
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = NBeatsConfig(learning_rate=args.lr)
    model = NBeatsModule(cfg)
    train_loader, val_loader = build_dataloaders(args.features, cfg, batch_size=args.batch_size)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        log_every_n_steps=25,
    )
    trainer.fit(model, train_loader, val_loader)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().eval()
    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
