"""Tests for backtest harness utility functions.

Focus on pure functions (determine_origins, seasonal_naive) to avoid heavy model dependency.
"""
import datetime as dt
from src.evaluation.backtest import determine_origins, seasonal_naive


def test_determine_origins_basic():
    # Create 60 sequential dates
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    # Convert to pandas Timestamps for function expectations
    import pandas as pd
    ts_dates = [pd.Timestamp(d) for d in dates]
    origins = determine_origins(ts_dates, horizon=5, stride=7, windows=4)
    assert len(origins) <= 4
    # Ensure chronological order
    assert origins == sorted(origins)
    # Each origin must have full horizon available in dates list
    last_day = ts_dates[-1]
    for o in origins:
        # Index difference check
        idx = ts_dates.index(o)
        assert idx + 5 <= len(ts_dates)


def test_seasonal_naive_fallback():
    import pandas as pd
    # History shorter than 7 days -> fallback to last value repeated
    history = pd.Series([10, 12, 13], index=[pd.Timestamp("2025-01-01") + pd.Timedelta(days=i) for i in range(3)])
    pred = seasonal_naive(history, horizon=6)
    assert pred == [13] * 6
    # Full weekly pattern case
    full_hist = pd.Series(range(1, 10), index=[pd.Timestamp("2025-02-01") + pd.Timedelta(days=i) for i in range(9)])
    pred2 = seasonal_naive(full_hist, horizon=10)
    assert len(pred2) == 10
    # Pattern repeats last 7 values
    pattern = list(full_hist.iloc[-7:])
    assert pred2[:7] == pattern