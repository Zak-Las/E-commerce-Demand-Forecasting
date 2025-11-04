"""Training script for N-BEATS minimal implementation.

Reads a panel parquet, constructs sliding window dataset, trains Lightning model,
and saves artifacts (model checkpoint + metrics summary JSON).

Example:
  python scripts/train_nbeats.py \
      --panel data/processed/m5_panel_subset.parquet \
      --input-length 112 --forecast-length 30 \
      --max-windows-per-item 20 --batch-size 32 --epochs 5

Artifacts:
  artifacts/models/nbeats_v{version}.ckpt
  artifacts/models/nbeats_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.models.nbeats_module import NBeatsModule, NBeatsConfig
from src.data.dataset_nbeats import PanelForecastDataset, PanelWindowConfig, split_dataset


def parse_args():
    ap = argparse.ArgumentParser(description="Train N-BEATS model")
    ap.add_argument("--panel", type=str, required=True, help="Path to panel parquet")
    ap.add_argument("--input-length", type=int, default=112)
    ap.add_argument("--forecast-length", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-stacks", type=int, default=4)
    ap.add_argument("--blocks-per-stack", type=int, default=3)
    ap.add_argument("--layer-width", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-windows-per-item", type=int, default=50)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel)
    cfg_ds = PanelWindowConfig(
        input_length=args.input_length,
        forecast_length=args.forecast_length,
        max_items=args.max_items,
        max_windows_per_item=args.max_windows_per_item,
    )
    dataset = PanelForecastDataset(panel_path, cfg_ds)
    train_ds, val_ds = split_dataset(dataset, val_fraction=args.val_fraction)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    cfg_model = NBeatsConfig(
        input_length=args.input_length,
        forecast_length=args.forecast_length,
        learning_rate=args.lr,
        num_stacks=args.num_stacks,
        num_blocks_per_stack=args.blocks_per_stack,
        layer_width=args.layer_width,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = NBeatsModule(cfg_model)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        accelerator="auto",
    )
    trainer.fit(model, train_loader, val_loader)

    # Save checkpoint manually
    ckpt_path = artifacts_dir / "nbeats_v0.1.0.ckpt"  # version can be templated later
    trainer.save_checkpoint(str(ckpt_path))

    # Collect final metrics from trainer logs (simplified)
    metrics = {
        "train_loss_final": float(trainer.callback_metrics.get("train_loss", -1)),
        "val_loss_final": float(trainer.callback_metrics.get("val_loss", -1)),
        "train_mae_final": float(trainer.callback_metrics.get("train_mae", -1)),
        "val_mae_final": float(trainer.callback_metrics.get("val_mae", -1)),
        "val_wape_final": float(trainer.callback_metrics.get("val_wape", -1)),
        "config": cfg_model.__dict__,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
    }
    with open(artifacts_dir / "nbeats_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved checkpoint:", ckpt_path)
    print("Saved metrics:", artifacts_dir / "nbeats_metrics.json")


if __name__ == "__main__":
    main()
