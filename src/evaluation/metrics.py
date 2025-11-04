"""Centralized metric functions for time series forecasting.

All metrics return floats; arrays converted via numpy.
WAPE definition: 100 * sum(|y - y_hat|) / sum(|y|) (NaN if all zeros).
SMAPE definition: 100 * mean(|y - y_hat| / ((|y| + |y_hat|)/2 + EPS)).

Usage:
    from src.evaluation.metrics import compute_all
    m = compute_all(y_true, y_pred)

This module unifies duplicated logic across baselines, backtests, and model evaluation.
"""
from __future__ import annotations

from typing import Dict, Iterable
import math
import numpy as np

EPS = 1e-8

def _to_array(x: Iterable) -> np.ndarray:
    return np.asarray(list(x) if not isinstance(x, (list, tuple, np.ndarray)) else x, dtype=float)

def rmse(y_true: Iterable, y_pred: Iterable) -> float:
    a = _to_array(y_true); b = _to_array(y_pred)
    return math.sqrt(float(np.mean((a - b) ** 2)))

def mae(y_true: Iterable, y_pred: Iterable) -> float:
    a = _to_array(y_true); b = _to_array(y_pred)
    return float(np.mean(np.abs(a - b)))

def mape(y_true: Iterable, y_pred: Iterable) -> float:
    a = _to_array(y_true); b = _to_array(y_pred)
    return 100.0 * float(np.mean(np.abs(a - b) / (np.abs(a) + EPS)))

def smape(y_true: Iterable, y_pred: Iterable) -> float:
    a = _to_array(y_true); b = _to_array(y_pred)
    denom = (np.abs(a) + np.abs(b)) / 2.0 + EPS
    return 100.0 * float(np.mean(np.abs(a - b) / denom))

def wape(y_true: Iterable, y_pred: Iterable) -> float:
    a = _to_array(y_true); b = _to_array(y_pred)
    denom = np.sum(np.abs(a))
    if denom == 0:
        return float("nan")
    return 100.0 * float(np.sum(np.abs(a - b)) / denom)

def forecast_accuracy_from_wape(wape_value: float) -> float:
    return float("nan") if np.isnan(wape_value) else 100.0 - wape_value

def compute_all(y_true: Iterable, y_pred: Iterable) -> Dict[str, float]:
    w = wape(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": w,
        "forecast_accuracy": forecast_accuracy_from_wape(w),
    }

__all__ = [
    "rmse","mae","mape","smape","wape","forecast_accuracy_from_wape","compute_all"
]
