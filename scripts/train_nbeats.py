"""Training script for N-BEATS (capstone primary model).

Enhancements over notebook prototype:
 - Optional YAML config loading (merges with CLI overrides).
 - Epoch metric recording (loss, WAPE, MAE, Accuracy).
 - Artifact versioning & lightweight registry integration.
 - Deterministic seed & reproducible dataset splits.

Usage (YAML config):
    python scripts/train_nbeats.py \
            --panel data/processed/m5_panel_subset.parquet \
            --config config/model/nbeats_v1.yaml

Or ad‑hoc overrides:
    python scripts/train_nbeats.py --panel data/processed/m5_panel_subset.parquet \
            --input-length 112 --forecast-length 30 --epochs 10 --layer-width 512

Artifacts (written to artifacts/models/):
    nbeats_{version}.ckpt            - checkpoint (Lightning save format)
    nbeats_{version}_metrics.json    - final + aggregate metrics
    registry.json                    - updated model registry (append entry if --register)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from src.models.nbeats_module import NBeatsModule, NBeatsConfig
from src.data.dataset_nbeats import PanelForecastDataset, PanelWindowConfig, split_dataset
from src.models.registry import Registry, ModelRecord


class EpochMetricsRecorder(pl.Callback):
    """Capture per-epoch metrics for later aggregation.

    Relies on metrics logged in the LightningModule; if absent stores NaNs.
    """
    def __init__(self):
        self.history = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401
        cm = trainer.callback_metrics
        entry = {
            "epoch": trainer.current_epoch + 1,
            "train_loss": float(cm.get("train_loss", float("nan"))),
            "val_loss": float(cm.get("val_loss", float("nan"))),
            "train_wape": float(cm.get("train_wape", float("nan"))),
            "val_wape": float(cm.get("val_wape", float("nan"))),
            "train_mae": float(cm.get("train_mae", float("nan"))),
            "val_mae": float(cm.get("val_mae", float("nan"))),
            "train_accuracy": float(cm.get("train_accuracy", float("nan"))),
            "val_accuracy": float(cm.get("val_accuracy", float("nan"))),
        }
        self.history.append(entry)


def parse_args():
    ap = argparse.ArgumentParser(description="Train N-BEATS model (capstone)")
    ap.add_argument("--panel", type=str, required=True, help="Path to item-aggregated panel parquet")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    # Overrides (optional)
    ap.add_argument("--input-length", type=int, default=None)
    ap.add_argument("--forecast-length", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-stacks", type=int, default=None)
    ap.add_argument("--blocks-per-stack", type=int, default=None)
    ap.add_argument("--layer-width", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--max-windows-per-item", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--version", type=str, default=None, help="Override model version tag")
    ap.add_argument("--register", action="store_true", help="Append trained model to registry.json")
    return ap.parse_args()


def load_yaml_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is not installed; cannot load YAML config")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_config(yaml_cfg: Dict[str, Any], args) -> Dict[str, Any]:
    """Merge CLI overrides onto YAML base, ignoring None overrides."""
    merged = dict(yaml_cfg)
    override_keys = [
        "input_length","forecast_length","batch_size","epochs","lr","num_stacks","blocks_per_stack",
        "layer_width","n_layers","dropout","max_items","max_windows_per_item","val_fraction","seed"
    ]
    for k in override_keys:
        cli_val = getattr(args, k.replace("blocks_per_stack","blocks_per_stack"), None)
        if cli_val is not None:
            # maintain naming differences between YAML and CLI
            merged[k if k != "blocks_per_stack" else "num_blocks_per_stack"] = cli_val if k != "blocks_per_stack" else cli_val
    # Normalize keys to model config names
    if "num_blocks_per_stack" not in merged and "blocks_per_stack" in merged:
        merged["num_blocks_per_stack"] = merged.pop("blocks_per_stack")
    return merged


def main():
    args = parse_args()
    yaml_cfg = load_yaml_config(args.config)
    merged = merge_config(yaml_cfg, args)

    # Extract canonical parameters with defaults if missing
    input_length = merged.get("input_length", 112)
    forecast_length = merged.get("forecast_length", 30)
    batch_size = merged.get("batch_size", 32)
    epochs = merged.get("epochs", 5)
    lr = merged.get("lr", merged.get("learning_rate", 1e-3))
    num_stacks = merged.get("num_stacks", 4)
    num_blocks_per_stack = merged.get("num_blocks_per_stack", 3)
    layer_width = merged.get("layer_width", 512)
    n_layers = merged.get("n_layers", 4)
    dropout = merged.get("dropout", 0.0)
    max_items = merged.get("max_items", None)
    max_windows_per_item = merged.get("max_windows_per_item", 50)
    val_fraction = merged.get("val_fraction", 0.1)
    seed = merged.get("seed", 42)
    version = args.version or yaml_cfg.get("version", "0.1.0")

    pl.seed_everything(seed, workers=True)
    start_time = time.time()

    artifacts_dir = Path("artifacts/models")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel parquet not found: {panel_path}")

    cfg_ds = PanelWindowConfig(
        input_length=input_length,
        forecast_length=forecast_length,
        max_items=max_items,
        max_windows_per_item=max_windows_per_item,
    )
    dataset = PanelForecastDataset(panel_path, cfg_ds)
    train_ds, val_ds = split_dataset(dataset, val_fraction=val_fraction)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    cfg_model = NBeatsConfig(
        input_length=input_length,
        forecast_length=forecast_length,
        learning_rate=lr,
        num_stacks=num_stacks,
        num_blocks_per_stack=num_blocks_per_stack,
        layer_width=layer_width,
        n_layers=n_layers,
        dropout=dropout,
    )
    model = NBeatsModule(cfg_model)

    recorder = EpochMetricsRecorder()
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        enable_checkpointing=False,
        accelerator="auto",
        callbacks=[recorder],
    )
    trainer.fit(model, train_loader, val_loader)

    # Save checkpoint
    ckpt_path = artifacts_dir / f"nbeats_{version}.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    # Final metrics
    cm = trainer.callback_metrics
    final_metrics = {
        "final_train_loss": float(cm.get("train_loss", float("nan"))),
        "final_val_loss": float(cm.get("val_loss", float("nan"))),
        "final_val_wape": float(cm.get("val_wape", float("nan"))),
        "final_val_mae": float(cm.get("val_mae", float("nan"))),
        "final_val_accuracy": float(cm.get("val_accuracy", float("nan"))),
    }
    # Aggregate (mean) across epochs
    if recorder.history:
        import pandas as pd  # local import to keep header light
        hist_df = pd.DataFrame(recorder.history)
        agg = {
            "mean_val_loss": float(hist_df.val_loss.mean()),
            "mean_val_wape": float(hist_df.val_wape.mean()),
            "mean_val_mae": float(hist_df.val_mae.mean()),
            "mean_val_accuracy": float(hist_df.val_accuracy.mean()),
        }
    else:
        agg = {"mean_val_loss": float("nan"), "mean_val_wape": float("nan"), "mean_val_mae": float("nan"), "mean_val_accuracy": float("nan")}

    metrics = {
        "version": version,
        "config": cfg_model.__dict__,
        "seed": seed,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "epochs": epochs,
        "runtime_seconds": round(time.time() - start_time, 2),
        "history": recorder.history,
        **final_metrics,
        **agg,
    }
    metrics_path = artifacts_dir / f"nbeats_{version}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved metrics    -> {metrics_path}")

    if args.register:
        reg = Registry(artifacts_dir / "registry.json")
        record = ModelRecord(
            name=f"nbeats_{version}",
            path=str(ckpt_path),
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metrics={
                "val_wape": metrics["final_val_wape"],
                "val_mae": metrics["final_val_mae"],
                "val_loss": metrics["final_val_loss"],
            },
            stage="candidate",
        )
        reg.append(record)
        print("Registry updated ->", reg.path)


if __name__ == "__main__":
    main()
