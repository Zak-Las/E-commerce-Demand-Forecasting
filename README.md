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
