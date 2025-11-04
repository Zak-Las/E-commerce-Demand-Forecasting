"""Baseline forecast evaluation for M5 subset.

Implements:
    - Naive (last value) baseline
    - Seasonal naive (weekly repeating pattern)

Refactored to use centralized metric functions from `src.evaluation.metrics` to avoid
duplication and maintain a single source of truth for RMSE, MAE, MAPE, sMAPE, WAPE, Accuracy.

Outputs:
    artifacts/metrics_naive_baseline.parquet
    artifacts/metrics_seasonal_naive_baseline.parquet
    artifacts/summary_naive_baselines.json

Usage:
    python -m src.evaluation.baselines \
            --panel data/processed/m5_panel_subset.parquet \
            --horizon 30 --top-k 25

Assumptions:
    - Panel parquet contains columns: item_id, date, demand (date convertible to datetime)
    - Aggregation to item-level demand already performed (store demand summed externally or in panel)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_all  # central metric utilities


def build_item_series(df: pd.DataFrame, item_id: str) -> pd.DataFrame:
    item = df[df["item_id"] == item_id]
    daily = item.groupby("date")["demand"].sum().reset_index().sort_values("date")
    return daily


def forecast_naive_last_value(history: pd.Series, horizon: int) -> np.ndarray:
    if len(history) == 0:
        return np.zeros(horizon)
    return np.full(horizon, history.iloc[-1])


def forecast_seasonal_weekly(history: pd.Series, horizon: int) -> np.ndarray:
    # Use last 7 values as repeating pattern; fallback to last value if < 7
    if len(history) < 7:
        return forecast_naive_last_value(history, horizon)
    pattern = history.iloc[-7:].values
    reps = int(np.ceil(horizon / 7))
    seq = np.tile(pattern, reps)[:horizon]
    return seq


def evaluate_baselines(panel_path: Path, horizon: int, top_k: int) -> Dict[str, Path]:
    df = pd.read_parquet(panel_path)
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])  # Ensure datetime
    df = df.sort_values("date")
    required = {"item_id", "date", "demand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    item_totals = df.groupby("item_id")["demand"].sum().sort_values(ascending=False)
    selected_items = item_totals.index[:top_k]

    naive_rows: List[Dict[str, float]] = []
    seasonal_rows: List[Dict[str, float]] = []

    for itm in selected_items:
        daily = build_item_series(df, itm)
        if len(daily) <= horizon:
            # Skip items without enough history
            continue
        train = daily.iloc[:-horizon]
        test = daily.iloc[-horizon:]
        actual = test["demand"].values

    naive_pred = forecast_naive_last_value(train["demand"], horizon)
    seasonal_pred = forecast_seasonal_weekly(train["demand"], horizon)

    m_naive = compute_all(actual, naive_pred); m_naive["item_id"] = itm
    m_seasonal = compute_all(actual, seasonal_pred); m_seasonal["item_id"] = itm
    naive_rows.append(m_naive)
    seasonal_rows.append(m_seasonal)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    naive_path = artifacts_dir / "metrics_naive_baseline.parquet"
    seasonal_path = artifacts_dir / "metrics_seasonal_naive_baseline.parquet"

    if naive_rows:
        pd.DataFrame(naive_rows).to_parquet(naive_path, index=False)
    if seasonal_rows:
        pd.DataFrame(seasonal_rows).to_parquet(seasonal_path, index=False)

    summary = {}
    if naive_rows:
        df_naive = pd.DataFrame(naive_rows)
        summary["naive_mean"] = df_naive[["rmse","mae","mape","smape","wape","forecast_accuracy"]].mean().to_dict()
        summary["naive_n_items"] = int(len(df_naive))
    if seasonal_rows:
        df_seasonal = pd.DataFrame(seasonal_rows)
        summary["seasonal_mean"] = df_seasonal[["rmse","mae","mape","smape","wape","forecast_accuracy"]].mean().to_dict()
        summary["seasonal_n_items"] = int(len(df_seasonal))

    summary_path = artifacts_dir / "summary_naive_baselines.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Naive baseline items:", summary.get("naive_n_items", 0))
    print("Seasonal naive baseline items:", summary.get("seasonal_n_items", 0))
    print("Summary saved to", summary_path)
    if summary:
        print(json.dumps(summary, indent=2))

    return {
        "naive": naive_path,
        "seasonal": seasonal_path,
        "summary": summary_path,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate naive baselines on M5 panel subset")
    ap.add_argument("--panel", type=str, required=True, help="Path to panel parquet (subset)")
    ap.add_argument("--horizon", type=int, default=28, help="Forecast horizon in days")
    ap.add_argument("--top-k", type=int, default=25, help="Number of top items by total demand to evaluate")
    return ap.parse_args()


def main():
    args = parse_args()
    panel_path = Path(args.panel)
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    evaluate_baselines(panel_path, args.horizon, args.top_k)


if __name__ == "__main__":
    main()
