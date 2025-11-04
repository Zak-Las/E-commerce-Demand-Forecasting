"""FastAPI service for forecasting.

Initial skeleton; will later wire model registry + N-BEATS/TFT inference.
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

app = FastAPI(title="E-commerce Demand Forecasting API", version="0.1.0")


class ForecastRequest(BaseModel):
    product_ids: List[str]
    horizon: int = 30
    model: Optional[str] = None  # 'prophet', 'nbeats', 'tft'


class ForecastPoint(BaseModel):
    product_id: str
    date: str
    forecast: float


class ForecastResponse(BaseModel):
    forecasts: List[ForecastPoint]
    model_used: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    # Placeholder: generate dummy deterministic forecasts until models wired.
    if req.horizon <= 0 or req.horizon > 90:
        raise HTTPException(status_code=400, detail="Horizon must be between 1 and 90")

    dates = pd.date_range(start=pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1), periods=req.horizon)
    forecasts: list[ForecastPoint] = []
    for pid in req.product_ids:
        # Simple seasonal-ish dummy pattern
        base = hash(pid) % 50 + 10
        for i, d in enumerate(dates):
            val = base * (1 + 0.05 * ((i % 7) - 3))
            forecasts.append(
                ForecastPoint(product_id=pid, date=d.date().isoformat(), forecast=float(round(val, 2)))
            )
    return ForecastResponse(forecasts=forecasts, model_used=req.model or "dummy")
