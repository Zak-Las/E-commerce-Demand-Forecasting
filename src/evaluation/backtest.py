"""Rolling-origin backtest harness for N-BEATS vs baselines.

Evaluates a trained N-BEATS checkpoint across multiple forecast origins (windows)
to assess stability of improvement vs seasonal naive baseline.

Simplifications (v1):
 - Reuse a single trained checkpoint (no retrain per window) for speed.
 - Per item forecast: take last `input_length` demands prior to origin and run model once.
 - Seasonal naive baseline: repeat last 7 days (weekly pattern) or fallback to last value if <7.
 - Aggregate metrics (mean) across items per window; also store per-window deltas.

Artifacts:
  artifacts/backtest/backtest_summary.json        (aggregate stats)
  artifacts/backtest/backtest_windows.parquet     (per-window metrics)

CLI:
  python -m src.evaluation.backtest \
      --panel data/processed/m5_panel_subset.parquet \
      --checkpoint artifacts/models/nbeats_0.1.0.ckpt \
      --horizon 30 --stride 7 --windows 12 --max-items 50

Exit if insufficient history to form requested number of windows.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch

from src.models.nbeats_module import NBeatsModule, NBeatsConfig
from src.evaluation.metrics import compute_all


@dataclass
class BacktestConfig:
    panel_path: Path
    checkpoint_path: Path
    horizon: int = 30
    stride: int = 7
    windows: int = 12
    input_length: int = 112  # must match training config
    max_items: int | None = 50
    artifacts_dir: Path = Path("artifacts/backtest")


def seasonal_naive(history: pd.Series, horizon: int) -> List[float]:
    if len(history) < 7:
        return [history.iloc[-1]] * horizon
    pattern = history.iloc[-7:].tolist()
    reps = (horizon + 6) // 7
    seq = (pattern * reps)[:horizon]
    return seq


def last_value_naive(history: pd.Series, horizon: int) -> List[float]:
    return [history.iloc[-1]] * horizon if len(history) else [0.0] * horizon


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"item_id", "date", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    return df


def determine_origins(unique_dates: List[pd.Timestamp], horizon: int, stride: int, windows: int) -> List[pd.Timestamp]:
    # origins must have full future horizon available within historical data
    origins = []
    for idx in range(len(unique_dates)):
        if idx + horizon <= len(unique_dates):
            origins.append(unique_dates[idx])
    # Take the last `windows` origins stepping backwards by `stride`
    selected: List[pd.Timestamp] = []
    current_index = len(origins) - 1 - horizon  # start near end minus horizon safeguard
    while current_index >= 0 and len(selected) < windows:
        selected.append(origins[current_index])
        current_index -= stride
    selected.reverse()  # chronological order
    return selected


def prepare_item_series(df: pd.DataFrame, itm: str) -> pd.Series:
    sub = df[df.item_id == itm].sort_values("date")
    return pd.Series(sub.demand.values, index=sub.date.values)


def model_forecast_single(model: NBeatsModule, series: pd.Series, origin: pd.Timestamp, input_length: int, horizon: int) -> List[float]:
    # Select history strictly before origin
    hist = series[series.index < origin]
    if len(hist) < input_length:
        raise ValueError("Insufficient history for required input_length")
    window = torch.tensor(hist.values[-input_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(window)
    return out.squeeze(0).tolist()[:horizon]


def run_backtest(cfg: BacktestConfig) -> Dict:
    df = load_panel(cfg.panel_path)
    items = df.item_id.unique().tolist()
    if cfg.max_items is not None:
        items = items[: cfg.max_items]

    unique_dates = sorted(df.date.unique().tolist())
    origins = determine_origins(unique_dates, cfg.horizon, cfg.stride, cfg.windows)
    if len(origins) == 0:
        raise RuntimeError("No valid origins determined; check horizon vs date range")

    # Load model checkpoint
    # Load checkpoint state_dict directly (handles earlier training script save format)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Reconstruct config from saved hyperparameters if available
    hparams = checkpoint.get("hyper_parameters", {})
    inferred_cfg = NBeatsConfig(
        input_length=hparams.get("input_length", cfg.input_length),
        forecast_length=hparams.get("forecast_length", cfg.horizon),
        num_stacks=hparams.get("num_stacks", 4),
        num_blocks_per_stack=hparams.get("num_blocks_per_stack", 3),
        n_layers=hparams.get("n_layers", 4),
        layer_width=hparams.get("layer_width", 512),
        learning_rate=hparams.get("learning_rate", 1e-3),
        dropout=hparams.get("dropout", 0.0),
    )
    model = NBeatsModule(inferred_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[backtest] Warning: missing keys:", len(missing))
    if unexpected:
        print("[backtest] Warning: unexpected keys:", len(unexpected))
    model.eval()

    per_window_rows: List[Dict] = []

    for origin in origins:
        window_metrics_rows: List[Dict] = []
        for itm in items:
            series = prepare_item_series(df, itm)
            try:
                y_pred = model_forecast_single(model, series, origin, cfg.input_length, cfg.horizon)
            except ValueError:
                continue  # skip items without enough history
            # Actual values: next horizon days starting at origin
            future_slice = series[(series.index >= origin)][: cfg.horizon]
            if len(future_slice) < cfg.horizon:
                continue  # incomplete horizon
            actual = future_slice.values.tolist()
            # Seasonal baseline
            baseline_pred = seasonal_naive(series[series.index < origin], cfg.horizon)
            m_model = compute_all(actual, y_pred)
            m_seasonal = compute_all(actual, baseline_pred)
            window_metrics_rows.append({
                "item_id": itm,
                "origin_date": origin.date().isoformat(),
                "model_wape": m_model["wape"],
                "baseline_wape": m_seasonal["wape"],
                "model_accuracy": m_model["forecast_accuracy"],
                "baseline_accuracy": m_seasonal["forecast_accuracy"],
            })
        if not window_metrics_rows:
            continue
        wdf = pd.DataFrame(window_metrics_rows)
        per_window_rows.append({
            "origin_date": origin.date().isoformat(),
            "n_items": int(len(wdf)),
            "model_wape_mean": float(wdf.model_wape.mean()),
            "baseline_wape_mean": float(wdf.baseline_wape.mean()),
            "delta_wape_mean": float(wdf.baseline_wape.mean() - wdf.model_wape.mean()),
            "model_accuracy_mean": float(wdf.model_accuracy.mean()),
            "baseline_accuracy_mean": float(wdf.baseline_accuracy.mean()),
        })

    if not per_window_rows:
        raise RuntimeError("Backtest produced no window metrics; may be insufficient history or filtering too strict.")

    result_df = pd.DataFrame(per_window_rows)
    summary = {
        "checkpoint": str(cfg.checkpoint_path),
        "panel": str(cfg.panel_path),
        "horizon": cfg.horizon,
        "stride": cfg.stride,
        "windows_requested": cfg.windows,
        "windows_evaluated": int(len(result_df)),
        "model_wape_mean": float(result_df.model_wape_mean.mean()),
        "baseline_wape_mean": float(result_df.baseline_wape_mean.mean()),
        "delta_wape_mean": float(result_df.delta_wape_mean.mean()),
        "model_accuracy_mean": float(result_df.model_accuracy_mean.mean()),
        "baseline_accuracy_mean": float(result_df.baseline_accuracy_mean.mean()),
        "per_window": per_window_rows,
    }

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path = cfg.artifacts_dir / "backtest_summary.json"
    parquet_path = cfg.artifacts_dir / "backtest_windows.parquet"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    result_df.to_parquet(parquet_path, index=False)
    print("Backtest summary ->", json_path)
    print("Per-window metrics ->", parquet_path)
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("_mean")}, indent=2))
    return summary


def parse_args():
    ap = argparse.ArgumentParser(description="Rolling-origin backtest for N-BEATS vs seasonal naive")
    ap.add_argument("--panel", type=Path, required=True, help="Panel parquet path (item_id,date,demand)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Trained N-BEATS checkpoint path")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--stride", type=int, default=7, help="Days between forecast origins")
    ap.add_argument("--windows", type=int, default=12, help="Number of origins to evaluate")
    ap.add_argument("--input-length", type=int, default=112, help="Model input length (must match training)")
    ap.add_argument("--max-items", type=int, default=50)
    return ap.parse_args()


def main():  # pragma: no cover
    args = parse_args()
    cfg = BacktestConfig(
        panel_path=args.panel,
        checkpoint_path=args.checkpoint,
        horizon=args.horizon,
        stride=args.stride,
        windows=args.windows,
        input_length=args.input_length,
        max_items=None if args.max_items <= 0 else args.max_items,
    )
    run_backtest(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
