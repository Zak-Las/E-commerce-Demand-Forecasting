# E-commerce Demand Forecasting

Predict the next 30 days of daily item-level demand using the M5 Forecasting dataset (primary) with optional Instacart enrichment. This capstone showcases depth in modern time series modeling (N-BEATS, later TFT), rigorous backtesting, and deployment via a FastAPI microservice.

## Why This Project Matters
Accurate short-term demand forecasts drive inventory optimization, reduced stockouts, and margin protection. Demonstrating a production-grade forecasting workflow highlights competency across data engineering, modeling, evaluation, and MLOps.

## Key Features (Planned)
- Reproducible data acquisition (`kaggle` API).  
- Feature engineering: lag features, rolling statistics, calendar & price signals.  
- Baseline models: Naive seasonal, moving average, Prophet.  
- Advanced deep model: N-BEATS (optimized), followed by Temporal Fusion Transformer.  
- Rolling-origin backtesting with multiple metrics (RMSE, MAPE, WAPE, sMAPE, pinball loss).  
- Lightweight model registry & promotion flow.  
- FastAPI service: `/forecast` returns JSON forecasts for requested product_ids.  
- Dockerized runtime for portable deployment.  

## Repository Structure (Evolving)
```
src/
	data/               # Download & loading scripts
	features/           # Feature engineering pipelines (TBD)
	models/             # Model wrappers & Lightning modules
	evaluation/         # Backtesting & metrics
	service/            # FastAPI application
notebooks/            # EDA & experimentation
docs/                 # Architecture, design notes, metrics artifacts
tests/                # Unit & smoke tests
data/ (gitignored)    # Raw & processed data (generated locally)
```

## Quick Start
1. Ensure you have Kaggle credentials (`~/.kaggle/kaggle.json`).  
2. Create & activate environment (Conda or Mamba recommended).  
3. Download M5 data:  
	 `python -m src.data.download_m5 --output data/raw/m5`  
4. (Upcoming) Run feature pipeline & train N-BEATS.  
5. Launch API (placeholder dummy forecasts until model wired):  
	 `uvicorn src.service.app:app --reload`  

## API (Initial Skeleton)
POST `/forecast`
```json
{
	"product_ids": ["FOO_001", "FOO_002"],
	"horizon": 30,
	"model": "nbeats"
}
```
Response (shape):
```json
{
	"model_used": "nbeats",
	"forecasts": [
		{"product_id": "FOO_001", "date": "2025-11-04", "forecast": 12.34}
	]
}
```

## Metrics (Planned)
| Metric | Purpose |
|--------|---------|
| RMSE | Penalize large errors |
| MAE | Robust central error |
| WAPE | Scale-aware for business impact |
| MAPE / sMAPE | Relative error (careful with zeros) |
| Pinball (q) | Probabilistic forecast quality |

### WAPE Definition & Rationale
Weighted Absolute Percentage Error (WAPE) measures total absolute forecast error relative to total actual demand:

$$\text{WAPE} = \frac{\sum_{i=1}^n |y_i - \hat{y}_i|}{\sum_{i=1}^n |y_i|} \times 100\%$$

Why it matters:
* Stable when many small or zero-demand days exist (MAPE can explode).
* Directly interpretable as "percent of volume mis-forecast".
* Portfolio-level primary KPI. We also report Forecast Accuracy = 100 - WAPE.

Difference vs MAPE:
* MAPE averages per-point percentage error; small denominators distort results.
* WAPE aggregates volume first, reducing volatility from intermittent products.

Edge case: If total actual demand is zero, WAPE is undefined (we emit `NaN`).

### Baseline Model Comparison (Latest Prophet Run)
Source: `artifacts/metrics_prophet_baseline.parquet` (25 items, 28-day horizon) & `artifacts/backtest_prophet_baseline.parquet` (5 rolling windows).

| Model | RMSE | MAE | WAPE (%) | sMAPE (%) | Forecast Accuracy (%) | Notes |
|-------|------|-----|----------|-----------|-----------------------|-------|
| Naive (last value) | 18.97 | 16.13 | 53.91 | 57.29 | 46.09 | Mean across 25 items |
| Seasonal Naive | 16.53 | 13.00 | 43.96 | 46.41 | 56.04 | Weekly pattern (last 7 days repeated) |
| Prophet (current avg) | 12.75 | 10.22 | 35.55 | 35.54 | 64.45 | Mean across 25 items |
| Prophet (single example) | 20.73 | 18.23 | 45.81 | 42.63 | 54.19 | Item: HOBBIES_1_178 |
| Prophet (backtest mean) | 26.80 | 21.20 | 46.93 | 44.60 | 53.07 | 5 rolling windows |
| N-BEATS (planned) | TBD | TBD | TBD | TBD | TBD | Deep residual stacks (next) |
| TFT (planned) | TBD | TBD | TBD | TBD | TBD | Multi-horizon with attention |

Notes:
* Forecast Accuracy = 100 - WAPE.
* Extremely large MAPE in backtest (3.27e9) indicates division-by-near-zero volatility; rely on WAPE/sMAPE for stability.
* Naive & Seasonal Naive baselines will be added to anchor improvement delta.

Historical table retained for planned models; update again after N-BEATS training.

## Modeling Focus
Depth-first: optimize N-BEATS (hyperparameters, early stopping, learning rate scheduling, gradient clipping, lagged feature variants). TFT added after stable N-BEATS benchmark.

## Roadmap Snapshot
1. Feature engineering module
2. N-BEATS Lightning implementation & training script
3. Backtesting harness & metrics report
4. Wire model registry + real forecasts in API
5. Add TFT & compare
6. Probabilistic / quantile forecasts
7. Instacart enrichment (optional advanced section)

See `docs/ARCHITECTURE.md` for deeper design details.

## License
TBD

## Contact
Author: Zak (Toronto) – Data Science & ML Engineering focus.
