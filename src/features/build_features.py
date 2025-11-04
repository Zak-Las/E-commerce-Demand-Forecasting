"""Feature engineering for M5 panel.

Reads partitioned Parquet files (item_id=xxx.parquet) and produces a training
tensor dataset for sequence models.

Simplifications (initial version):
- Use last N days for input window and next forecast_length days as target.
- Only basic lags & rolling means engineered inline.

Future enhancements:
- Price elasticity features, event flags, per-item seasonality indices.
- Probabilistic targets (quantiles) or hierarchical reconciliation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import argparse
import sys


@dataclass
class FeatureConfig:
    input_length: int = 28 * 4
    forecast_length: int = 30
    min_history: int = 400  # require at least this many days to keep item
    data_dir: Path = Path("data/processed/m5_panel")
    max_items: int | None = 500  # limit number of items to reduce memory; None means all
    stride: int = 1  # sampling stride across windows
    save_path: Path = Path("data/processed/features.npz")


def load_item(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure sorted by date
    df = df.sort_values("date")
    return df


def build_features_for_item(df: pd.DataFrame, cfg: FeatureConfig):
    if len(df) < cfg.min_history + cfg.forecast_length:
        return None
    # Basic demand series
    y = df["demand"].astype(float).to_numpy()
    # Lags: 1,7,14,28
    lags = {}
    for lag in [1, 7, 14, 28]:
        arr = np.concatenate([np.full(lag, np.nan), y[:-lag]])
        lags[f"lag_{lag}"] = arr
    # Rolling means
    def rolling_mean(a, w):
        out = np.full_like(a, np.nan, dtype=float)
        if len(a) >= w:
            cumsum = np.cumsum(np.insert(a, 0, 0))
            out[w - 1 :] = (cumsum[w:] - cumsum[:-w]) / w
        return out

    roll_7 = rolling_mean(y, 7)
    roll_28 = rolling_mean(y, 28)

    feature_matrix = np.vstack([
        y,
        lags["lag_1"],
        lags["lag_7"],
        lags["lag_14"],
        lags["lag_28"],
        roll_7,
        roll_28,
    ])  # (n_features, T)

    # Build supervised samples via sliding window
    samples_X = []
    samples_Y = []
    T = feature_matrix.shape[1]
    for end in range(cfg.input_length, T - cfg.forecast_length, cfg.stride):
        window_feats = feature_matrix[:, end - cfg.input_length : end]
        target_seq = y[end : end + cfg.forecast_length]
        if np.isnan(window_feats).any():
            continue
        samples_X.append(window_feats)
        samples_Y.append(target_seq)

    if not samples_X:
        return None
    X = np.stack(samples_X)  # (N, n_features, input_length)
    Y = np.stack(samples_Y)  # (N, forecast_length)
    # Flatten feature dimension for baseline N-BEATS design expecting (batch, input_length)
    X_flat = X.reshape(X.shape[0], -1)
    return torch.tensor(X_flat, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def build_dataset(cfg: FeatureConfig):
    paths = list(cfg.data_dir.glob("item_id=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No partition files found in {cfg.data_dir}. Run prepare_m5.py first.")
    if cfg.max_items is not None and len(paths) > cfg.max_items:
        paths = paths[: cfg.max_items]
    all_X = []
    all_Y = []
    for idx, p in enumerate(paths, start=1):
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(paths)} items...")
        df = load_item(p)
        out = build_features_for_item(df, cfg)
        if out is None:
            continue
        X, Y = out
        all_X.append(X)
        all_Y.append(Y)
    if not all_X:
        raise RuntimeError(
            "No samples produced; consider lowering min_history or increasing max_items/stride adjustments."
        )
    X_cat = torch.cat(all_X, dim=0)
    Y_cat = torch.cat(all_Y, dim=0)
    return X_cat, Y_cat


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build feature tensors for M5")
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/m5_panel"))
    ap.add_argument("--input-length", type=int, default=28 * 4)
    ap.add_argument("--forecast-length", type=int, default=30)
    ap.add_argument("--min-history", type=int, default=400)
    ap.add_argument("--max-items", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--save-path", type=Path, default=Path("data/processed/features.npz"))
    return ap.parse_args(argv or sys.argv[1:])


def main():  # pragma: no cover
    args = parse_args()
    cfg = FeatureConfig(
        input_length=args.input_length,
        forecast_length=args.forecast_length,
        min_history=args.min_history,
        data_dir=args.data_dir,
        max_items=None if args.max_items <= 0 else args.max_items,
        stride=args.stride,
        save_path=args.save_path,
    )
    X, Y = build_dataset(cfg)
    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.save_path, X=X.numpy(), Y=Y.numpy())
    print("Saved features to", cfg.save_path, "X shape", X.shape, "Y shape", Y.shape)


if __name__ == "__main__":
    main()
