# Architecture Overview

## Goals
- Predict 30-day daily demand for retail items (M5 primary dataset) with high fidelity.
- Provide an API for on-demand forecasts per product.
- Showcase depth on a single advanced model (optimize N-BEATS first), add TFT second.

## High-Level Flow
```mermaid
flowchart LR
  A[Download Raw Data\n(Kaggle API)] --> B[Staging Layer\n data/raw/m5]
  B --> C[Processing & Normalization\n Parquet Partitioning]
  C --> D[Feature Engineering\n lags, rolling stats, calendar]
  D --> E[Training Dataset\n temporal splits]
  E --> F[N-BEATS Training\n PyTorch Lightning]
  F --> G[Model Registry\n metadata.json]
  G --> H[FastAPI Service]\n
  H -->|/forecast| I[Client]
```

## Components
### Data Layer
- Script: `src/data/download_m5.py` handles pulling competition files.
- Future: `src/data/instacart_*` for optional enrichment.
- Store processed features under `data/processed/` (gitignored) as Parquet.

### Feature Engineering
- `src/features/build_features.py` (TBD) constructs lag features (1,7,14,28), rolling mean/median windows, price change features, calendar events.

### Modeling
- Baseline: Prophet wrapper (`src/models/baseline_prophet.py`).
- Core: N-BEATS LightningModule (to implement) with configurable stack/block widths.
- Secondary: TFT (later) leveraging static categorical embeddings (store, dept, item) and known future inputs (calendar features).

### Evaluation
- Rolling-origin backtest utility will produce metrics & a leaderboard artifact (`docs/metrics/latest.json`).

### Service
- FastAPI app (`src/service/app.py`) loads the latest production model on startup and serves forecasts.
- Inference abstraction to map product_ids -> internal item indices.

### Configuration
- Simple `config/` (to add) with YAML for model hyperparameters and paths.

## Model Registry (Lightweight)
A JSON file `models/registry.json` containing entries like:
```json
[
  {"name": "nbeats_v1", "path": "models/nbeats_v1.pt", "created": "2025-11-03T12:00:00Z", "metrics": {"RMSE": 3.42}, "stage": "production"}
]
```
Upgrade later to MLflow if desired.

## Backtesting Strategy
- Use last 90 days as evaluation horizon across rolling windows of size 30 (stride 7) to capture stability.
- Metrics aggregated (mean, std) for each model.

## Performance Considerations
- M1 Pro: Use PyTorch MPS backend; ensure device selection logic (`mps` if available else `cpu`).
- Keep batch sizes moderate; N-BEATS benefits from gradient clipping + learning rate finder.

## Security / Reliability
- Input validation on API (max product ids per request, horizon limits).
- Deterministic seeds for reproducibility.

## Roadmap Snapshot
1. Implement feature builder.
2. Implement N-BEATS Lightning module + trainer script.
3. Add backtesting harness.
4. Wire model registry & load in FastAPI.
5. Add TFT implementation.
6. Optional: Quantile forecasts (pinball loss) + prediction intervals.
