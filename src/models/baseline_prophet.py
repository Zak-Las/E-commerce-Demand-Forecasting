"""Baseline Prophet model wrapper.

We keep a thin abstraction so we can later slot in N-BEATS/TFT with a common interface.

NOTE: Prophet install (cmdstan) is heavier; decide later if to include or switch to statsmodels SARIMAX
for lighter footprint. For now we provide a stub to be expanded.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

try:
    from prophet import Prophet  # type: ignore
except Exception:  # pragma: no cover - optional dependency early stage
    Prophet = None  # type: ignore


@dataclass
class ProphetConfig:
    daily_seasonality: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    changepoint_prior_scale: float = 0.05


class ProphetBaseline:
    def __init__(self, config: ProphetConfig | None = None):
        self.config = config or ProphetConfig()
        if Prophet is None:
            raise ImportError("prophet package not installed. Add it to environment or skip this baseline.")
        self.model = Prophet(
            daily_seasonality=self.config.daily_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            yearly_seasonality=self.config.yearly_seasonality,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "ProphetBaseline":
        """Fit on a dataframe with columns: ds (datetime), y (float)."""
        self.model.fit(df)
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit before forecast")
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        return forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={"yhat": "forecast"}
        )
