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

## Reproducibility & One-Command Pipeline
This project treats notebooks as exploratory artifacts only—every production / evaluation step is scriptable. Recruiters and reviewers can reproduce core results without manual cell execution.

### Deterministic Seeds
All training scripts set a fixed seed (default 42) via PyTorch Lightning to keep splits and weight initialization stable. Override with `--seed` if needed.

### Data Lineage
Raw M5 CSVs are downloaded with `download-m5` and never modified. Processed panel Parquet files live under `data/processed/` (gitignored). Feature tensors and model artifacts live under `artifacts/` with versioned filenames.

### Makefile Targets
```
make download-m5      # Fetch raw M5 competition files
make train-nbeats     # Run N-BEATS training using scripts/train_nbeats.py (config-driven)
make test             # Execute unit & smoke tests (will expand as suite grows)
make lint             # Static analysis via ruff
make api              # Start FastAPI (dummy forecasts until model wired)
```

### Recommended Reproduction Path (Current Phase)
1. Download data: `make download-m5`
2. Prepare long panel (partitioned): `python -m src.data.prepare_m5 --raw-dir data/raw/m5 --out-dir data/processed/m5_panel`
3. (Optional) Build engineered features: `python -m src.features.build_features` (will evolve)
4. Train N-BEATS: `make train-nbeats` or specify config `python scripts/train_nbeats.py --panel data/processed/m5_panel_subset.parquet --config config/model/nbeats_v1.yaml --register`
5. Run baselines: `python -m src.evaluation.baselines --panel data/processed/m5_panel_subset.parquet --horizon 30 --top-k 25`
6. (Upcoming) Backtest harness: `python -m src.evaluation.backtest --panel ... --checkpoint artifacts/models/nbeats_0.1.0.ckpt --horizon 30 --stride 7 --windows 12`
7. Inspect artifacts: `artifacts/models/*.ckpt`, `*_metrics.json`, baseline parquet summaries.

### Planned Single Command (Future)
`make reproduce` will orchestrate: download → prepare → features → baselines → train → backtest → generate leaderboard.

### Model Registry & Promotion
`scripts/train_nbeats.py --register` appends metadata to `artifacts/models/registry.json` (stage=candidate). A future promotion command will flip a candidate to production (for service loading) while archiving previous prod entries.

### Why This Matters
Explicit scripted path + versioned artifacts demonstrates engineering rigor: hiring managers can audit each stage, diff config changes, and verify claimed metric improvements (e.g., ≥8 WAPE point gain vs seasonal naive).

### Integrity & Hashing (Planned)
Panel subsets and feature exports will include a SHA256 hash in a sidecar JSON for provenance, ensuring reproducibility even if intermediate files are regenerated.

### Notebook Usage Policy
Exploration only: training and evaluation logic lives in `src/` & `scripts/`. Final results visualization notebook will consume artifacts, not produce them.

---

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

## Versioning & Releases
This project uses Semantic Versioning (`MAJOR.MINOR.PATCH`) and documents changes in `CHANGELOG.md` (Keep a Changelog format).

Current version: `0.1.0`.

### Release Workflow
1. Develop features on topic branches (e.g., `feat/nbeats-training`).
2. Update `CHANGELOG.md` under `Unreleased` with notable additions/fixes.
3. When cutting a release:
	- Move entries from `Unreleased` to a new version section with date.
	- Bump `__version__` in `src/__init__.py` and `version` in `pyproject.toml`.
4. Tag the release (after merging to `main`):
	```bash
	git checkout main
	git pull origin main
	git tag -a v0.1.0 -m "Baseline evaluation foundation"
	git push origin v0.1.0
	```
5. Create a GitHub Release referencing the tag (include metrics deltas and highlights).

### Version Bumping Guidelines
- Patch (`0.1.1`): Non-breaking fixes, internal refactors, doc updates.
- Minor (`0.2.0`): New models, endpoints, new metrics.
- Major (`1.0.0`): Stable API surface, production deployment, breaking changes.

### Planned Automation
- CI check to ensure tag matches `__version__` and `pyproject.toml`.
- Changelog diff generation in release PR template.
