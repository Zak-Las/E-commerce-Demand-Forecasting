# Lean E-commerce Demand Forecasting

Predict the next 30 days of daily item-level demand (M5 subset or synthetic) with a single deep learning model (N-BEATS). Repository is intentionally minimal to showcase rapid prototyping, evaluation, and clear communication — ideal for portfolio & role alignment.

## Included
* Sliding window dataset builder: `src/data/dataset_nbeats.py`
* Lightweight N-BEATS implementation: `src/models/nbeats_module.py`
* Central metric functions (WAPE, MAE, etc.): `src/evaluation/metrics.py`
* End-to-end training & analysis notebook: `notebooks/nbeats_training.ipynb`

Removed for simplicity: API service, Dockerfile, Makefile, registry, heavy backtest harness, production infra.

## Quick Start
1. Create environment (example):
	 ```bash
	 conda env create -f environment.yml  # or: pip install -e .
	 conda activate demand-forecast
	 ```
2. Ensure panel parquet at `data/processed/m5_panel_subset.parquet` (columns: `item_id,date,demand`). If absent, notebook synthesizes data.
3. Open `notebooks/nbeats_training.ipynb` → Run all.
4. Compare model WAPE vs seasonal naive (repeat last 7 days). Iterate.

## Core Metric
WAPE (Weighted Absolute Percentage Error):
```
WAPE = sum(|y - ŷ|) / sum(|y|) * 100
Accuracy = 100 - WAPE
```
Baseline: seasonal naive (weekly repeat). Goal: show disciplined evaluation even if improvement is modest.

## Relevance to D.S. Roles
This lean forecasting project evidences the competencies requested in the GHGSat posting:
| D.S. Responsibility / Attribute | Evidence in Repo |
| --------------------------------- | ---------------- |
| Data Exploration & Curation | Notebook EDA steps (panel inspection, scaling), sliding window construction logic |
| Data Quality Mindset | Planned Phase 1: add `data_quality.ipynb` (missing dates, duplicates, zero-demand streaks) |
| Rapid Prototyping | Single notebook trains & evaluates N-BEATS + baseline within minutes on CPU |
| Model Design & Validation | Custom N-BEATS module + seasonal naive comparison + mini rolling-origin preview cell |
| Metric & Error Analysis | Centralized metrics module, WAPE rationale, per-batch aggregation in notebook |
| Communicating Insight | Model card generation cell (artifacts: metrics JSON + markdown) |
| Iterative Improvement | Roadmap: calendar & lag features, per-item scaling, anomaly detection prototype |
| Impact Orientation | Focus on interpretable accuracy delta vs naive baseline (business-aligned) |
| Continuous Learning | Clear next-step plan (feature engineering, probabilistic forecasts) |

Use this mapping directly in applications to highlight immediate role fit (time-series rigor → emissions signal interpretation, anomaly detection, and scalable analytics).

## Simple Next Experiments
1. Calendar features (weekday, month). 
2. Lag features (lag_7, rolling_mean_7). 
3. Per-item vs global scaling comparison. 
4. Early stopping + learning rate schedule.

## Data Assumptions
Daily demand, dense panel (one row per item/date). If gaps exist, forward-fill or impute before windowing (not yet automated).

## Layout
```
src/
	data/dataset_nbeats.py
	models/nbeats_module.py
	evaluation/metrics.py
notebooks/
	nbeats_training.ipynb
data/processed/        # (gitignored)
artifacts/models/      # notebook outputs (gitignored by default)
```

## Why Keep It Lean?
Reduces cognitive load for reviewers; centers on signal: data prep → model → baseline → metrics → artifact. Infrastructure can be reintroduced later without obscuring core skill demonstration.

## Future (Optional) Extensions
* Feature engineering module
* Lightweight anomaly detection (emissions-analog bridging)
* Rolling-origin harness script
* Simple inference API

## Author
Zak – Toronto

## License
TBD

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

## Backtesting (Rolling-Origin Evaluation)
To demonstrate stability (not just a lucky single split), the project runs a rolling-origin backtest using the harness in `src/evaluation/backtest.py`.

### Why Rolling-Origin?
Traditional single validation splits can overstate performance if the chosen segment has favorable seasonality or demand levels. Rolling-origin evaluates multiple forecast start dates ("origins") and aggregates metrics:
- Robustness: Mean + variance of WAPE across windows
- Improvement Consistency: Average ΔWAPE vs seasonal naive baseline
- Drift Sensitivity: Detect degradation if recent windows worsen

### Procedure (v1)
1. Select last N chronological origins (default N=12) separated by a stride (default 7 days).
2. For each origin, use all available history strictly before origin.
3. Forecast next 30 days with the trained N-BEATS checkpoint (no retraining per window in v1 for speed).
4. Compute metrics (RMSE, MAE, WAPE, sMAPE, Accuracy) using `src.evaluation.metrics`.
5. Compute seasonal naive baseline (repeat last 7 days) for comparison.
6. Persist summary JSON + per-window parquet in `artifacts/backtest/`.

### Command
```
python -m src.evaluation.backtest \
		--panel data/processed/m5_panel_subset.parquet \
		--checkpoint artifacts/models/nbeats_0.1.0.ckpt \
		--horizon 30 --stride 7 --windows 12 --max-items 50
```

### Artifacts
```
artifacts/backtest/backtest_summary.json      # aggregate means & deltas
artifacts/backtest/backtest_windows.parquet   # per-origin metrics
```

Example (placeholder → will update after first run):
```jsonc
{
	"horizon": 30,
	"stride": 7,
	"windows_evaluated": 12,
	"model_wape_mean": "TBD",
	"baseline_wape_mean": "TBD",
	"delta_wape_mean": "TBD (target ≥ 8)",
	"model_accuracy_mean": "TBD"
}
```

### Interpreting Results
- Primary success signal: Mean ΔWAPE ≥ 8 percentage points vs seasonal naive.
- Stability heuristic: Std(WAPE) / Mean(WAPE) ≤ ~0.15.
- If ΔWAPE deteriorates in latest 3 windows, schedule retraining or investigate drift.

### Future Enhancements
- Optional per-window retraining flag (`--retrain-per-window`) to test robustness of training procedure (deferred).
- Quantile evaluation (pinball loss) per window for probabilistic forecasts.
- Automatic leaderboard generation merging baseline + N-BEATS + future TFT.

After first successful run, numbers will replace the placeholders above and feed into the N-BEATS model card.

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
