"""Parametric tests for metric functions used in baseline evaluation."""
import math
import numpy as np
import pytest

from src.evaluation.baselines import rmse, mae, mape, smape, wape, compute_metrics


# RMSE cases: perfect prediction, simple error pattern, negative values
@pytest.mark.parametrize(
    "actual,pred,expected",
    [
        ([1, 2, 3], [1, 2, 3], 0.0),
        ([1, 2, 3], [2, 2, 2], math.sqrt(((1)**2 + 0**2 + (1)**2)/3)),
        ([-1, -2], [-1, -1], math.sqrt((0**2 + 1**2)/2)),
    ],
)
def test_rmse_parametric(actual, pred, expected):
    assert math.isclose(rmse(actual, pred), expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "actual,pred,expected",
    [
        ([1, 2, 3], [1, 2, 3], 0.0),
        ([1, 2, 3], [2, 2, 2], (1 + 0 + 1) / 3),
        ([-1, -2], [-1, -1], (0 + 1) / 2),
    ],
)
def test_mae_parametric(actual, pred, expected):
    assert math.isclose(mae(actual, pred), expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "actual,pred,expected",
    [
        ([10, 20], [10, 20], 0.0),
        ([10, 20], [11, 18], 100 * ((1/10 + 2/20) / 2)),  # (0.1 + 0.1)/2 *100
        ([5, 5, 10], [4, 6, 11], 100 * np.mean([1/5, 1/5, 1/10])),
    ],
)
def test_mape_parametric(actual, pred, expected):
    # Use absolute tolerance due to floating point accumulation with EPS handling
    assert math.isclose(mape(actual, pred), expected, rel_tol=1e-7, abs_tol=1e-7)


@pytest.mark.parametrize(
    "actual,pred,expected",
    [
        ([10, 20], [10, 20], 0.0),
        ([10, 20], [11, 18], 100 * np.mean([
            abs(1) / ((10 + 11) / 2),
            abs(2) / ((20 + 18) / 2),
        ])),
        ([5, 5, 10], [4, 6, 11], 100 * np.mean([
            1/((5+4)/2),
            1/((5+6)/2),
            1/((10+11)/2),
        ])),
    ],
)
def test_smape_parametric(actual, pred, expected):
    assert math.isclose(smape(actual, pred), expected, rel_tol=1e-7, abs_tol=1e-7)


@pytest.mark.parametrize(
    "actual,pred,expected,is_nan",
    [
        ([10, 20], [10, 20], 0.0, False),  # perfect
        ([10, 20], [11, 18], 100 * (1 + 2) / (10 + 20), False),
        ([0, 0, 0], [1, 2, 3], float('nan'), True),  # all-zero actual -> NaN
        ([0, 5, 0, 10], [0, 6, 1, 12], 100 * (0 + 1 + 1 + 2) / (0 + 5 + 0 + 10), False),
    ],
)
def test_wape_parametric(actual, pred, expected, is_nan):
    val = wape(actual, pred)
    if is_nan:
        assert math.isnan(val)
    else:
        assert math.isclose(val, expected, rel_tol=1e-9)


@pytest.mark.parametrize(
    "actual,pred",
    [
        ([10, 20], [10, 20]),
        ([10, 20], [11, 18]),
        ([0, 0, 0], [1, 2, 3]),
        ([0, 5, 0, 10], [0, 6, 1, 12]),
    ],
)
def test_compute_metrics_accuracy_relation(actual, pred):
    metrics = compute_metrics("ITEM_X", np.array(actual), np.array(pred))
    w = metrics["wape"]
    acc = metrics["forecast_accuracy"]
    if math.isnan(w):
        assert math.isnan(acc)
    else:
        assert math.isclose(acc, 100.0 - w, rel_tol=1e-9)
    # Ensure essential keys exist
    for key in ["rmse", "mae", "mape", "smape", "wape", "forecast_accuracy"]:
        assert key in metrics
