import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.baselines import (
    rmse,
    mae,
    mape,
    smape,
    wape,
    forecast_naive_last_value,
    forecast_seasonal_weekly,
    compute_metrics,
    evaluate_baselines,
)


def test_metrics_basic():
    actual = [10, 20, 30]
    pred = [11, 19, 33]
    assert math.isclose(rmse(actual, pred), math.sqrt(((1**2)+(1**2)+(3**2))/3), rel_tol=1e-6)
    assert math.isclose(mae(actual, pred), (1+1+3)/3, rel_tol=1e-6)
    # MAPE ~ average(|err|/actual)
    expected_mape = 100 * np.mean([1/10, 1/20, 3/30])
    assert math.isclose(mape(actual, pred), expected_mape, rel_tol=1e-6)
    # sMAPE
    smape_terms = [abs(1)/( (10+11)/2 ), abs(1)/( (20+19)/2 ), abs(3)/( (30+33)/2 )]
    expected_smape = 100 * np.mean(smape_terms)
    assert math.isclose(smape(actual, pred), expected_smape, rel_tol=1e-6)
    # WAPE
    expected_wape = 100 * (1+1+3)/(10+20+30)
    assert math.isclose(wape(actual, pred), expected_wape, rel_tol=1e-6)


def test_wape_zero_actual():
    actual = [0, 0, 0]
    pred = [1, 2, 3]
    assert math.isnan(wape(actual, pred)), "WAPE should be NaN when all actuals are zero"


def test_naive_forecast_last_value():
    history = pd.Series([5, 6, 7])
    horizon = 4
    fc = forecast_naive_last_value(history, horizon)
    assert len(fc) == horizon
    assert all(v == 7 for v in fc), "Naive forecast should repeat last value"


def test_seasonal_weekly_forecast_pattern():
    pattern = list(range(10, 17))  # 7 values
    history = pd.Series(pattern)
    horizon = 10
    fc = forecast_seasonal_weekly(history, horizon)
    assert len(fc) == horizon
    # First 7 should equal last 7 of history
    assert np.array_equal(fc[:7], np.array(pattern)), "Seasonal forecast first 7 days should match pattern"
    # Remainder should continue pattern
    assert np.array_equal(fc[7:], np.array(pattern[:3]))


def test_compute_metrics_forecast_accuracy():
    actual = np.array([10, 20])
    pred = np.array([12, 18])  # abs errors 2 + 2 = 4; total actual 30 => WAPE=13.333...
    metrics = compute_metrics("ITEM_A", actual, pred)
    expected_wape = 100 * (4/30)
    assert math.isclose(metrics["wape"], expected_wape, rel_tol=1e-6)
    assert math.isclose(metrics["forecast_accuracy"], 100 - expected_wape, rel_tol=1e-6)


def test_evaluate_baselines_integration(tmp_path):
    # Synthetic panel with two items over 40 days
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    data_rows = []
    for d in dates:
        # ITEM1: linear growth
        data_rows.append({"item_id": "ITEM1", "date": d, "demand": (d - dates[0]).days + 1})
        # ITEM2: weekly seasonality  (mod 7 pattern)
        data_rows.append({"item_id": "ITEM2", "date": d, "demand": ((d - dates[0]).days % 7) + 1})
    panel_df = pd.DataFrame(data_rows)
    panel_path = tmp_path / "panel.parquet"
    panel_df.to_parquet(panel_path, index=False)

    artifacts = evaluate_baselines(panel_path, horizon=7, top_k=2)
    # Ensure artifact files exist
    assert artifacts["naive"].exists(), "Naive metrics parquet missing"
    assert artifacts["seasonal"].exists(), "Seasonal naive metrics parquet missing"
    naive_df = pd.read_parquet(artifacts["naive"])
    seasonal_df = pd.read_parquet(artifacts["seasonal"])
    required_cols = {"item_id", "rmse", "mae", "mape", "smape", "wape", "forecast_accuracy"}
    assert required_cols.issubset(set(naive_df.columns))
    assert required_cols.issubset(set(seasonal_df.columns))
    # Each item should have one metrics row
    assert len(naive_df) <= 2 and len(seasonal_df) <= 2


def test_seasonal_weekly_fallback_short_history():
    """History length < 7 should fallback to naive last value repetition."""
    short_hist = pd.Series([2, 4, 6])  # len=3 < 7
    horizon = 5
    fc = forecast_seasonal_weekly(short_hist, horizon)
    assert len(fc) == horizon
    assert np.all(fc == 6), "Forecast should repeat last value for short history"


def test_wape_mixed_zero_actuals():
    actual = [0, 5, 0, 10]
    pred = [0, 6, 1, 12]  # errors: 0,1,1,2 => sum=4; denom=sum(actual)=15 => wape=26.666...
    expected = 100 * (4/15)
    assert math.isclose(wape(actual, pred), expected, rel_tol=1e-6)


def test_compute_metrics_all_zero_actuals():
    actual = np.zeros(4)
    pred = np.array([1, 2, 0, 3])
    metrics = compute_metrics("ITEM_ZERO", actual, pred)
    assert math.isnan(metrics["wape"])
    assert math.isnan(metrics["forecast_accuracy"])


if __name__ == "__main__":
    import pytest  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
