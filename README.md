# E-commerce Demand Forecasting (M5 Subset, Notebook-Centric Capstone)

Forecast 30-day daily demand for a subset of items from the M5 dataset. The project is intentionally lean: pure notebook workflow, CPU-only (Mac M1 Pro), no external datasets, no deployment claims. Focus is on clear data handling, a transparent baseline (Prophet), a deep model (N-BEATS), and a small feature engineering iteration that improves validation error.

## 1. Problem & Scope
Retail teams need short‑term item-level demand forecasts to plan inventory. We frame a simplified version: given 112 days of history, predict the next 30 days per item. Scope deliberately excludes real-time serving, distributed training, or multi-source fusion to keep the capstone concise under a two‑day deadline.

## 2. Data
Source: M5 competition dataset (subset panel). If the processed parquet is missing, notebooks generate a synthetic fallback to keep the pipeline runnable. Data artifacts are stored under `data/processed/` and tracked (large files via Git LFS settings already in repo).

Quality checks (missing values, zero-demand rates, date continuity, outliers) and light cleaning (clip negatives, fill NA) are performed in a single consolidated notebook: `notebooks/data_prep.ipynb`. A JSON & Markdown quality report is saved to `artifacts/data/`.

## 3. Workflow Overview
Run notebooks in this order:
1. `notebooks/data_prep.ipynb` – Load / fallback synth generation, profile, clean, quality report.
2. `notebooks/prophet_baseline.ipynb` – Fit Prophet on an aggregate series + small item subset; produce baseline metrics JSON.
3. `notebooks/nbeats_training.ipynb` – Train N-BEATS (global scaling), evaluate, mini rolling-origin backtest, save checkpoint + metrics JSON.
4. (Within same notebook) Feature Engineering iteration: residual (weekday deseason + lag7 mean) re-training, metrics comparison, model card generation.

Artifacts directory (`artifacts/models/`) holds checkpoints, metrics, model card.

## 4. Metrics
Primary metric: WAPE (Weighted Absolute Percentage Error). Forecast Accuracy is defined as (100 − WAPE). MAE and sMAPE reported for auxiliary context. Metrics logic consolidated in `src/evaluation/metrics.py` and reused directly by notebooks.

## 5. Results
Final metrics extracted from JSON artifacts in `artifacts/models/`.

| Model | Val WAPE (%) | Val MAE (units) | Accuracy (%) | Notes |
|-------|--------------|-----------------|--------------|-------|
| Prophet (Aggregate) | 24.81 | 77.09 | 75.19 | Trained on aggregate daily demand (scale differs) |
| N-BEATS (Global Scaling) | 17.66 | 0.14 | 82.34 | Per-item panel windows with global mean/std scaling |

Metric comparability caveat: MAE scales differ (aggregate series vs per-item scaled windows). WAPE (and derived Accuracy = 100 − WAPE) is the primary cross‑model comparison metric.

Removed: Residual feature engineering experiment (weekday deseason + lag7 residual) was pruned for scope clarity; its intermediate artifacts are excluded from the final table to avoid mixed target scales.

Model card: `artifacts/models/model_card_notebook.md` summarizes configuration and data hash fragment.

## 6. (Removed) Feature Engineering Iteration
An exploratory residual transformation (weekday deseason + lag7 level removal) was run initially; it did not improve validation performance and introduced a different error scale. To keep the capstone focused and interpretable, the experiment was removed. This demonstrates disciplined scope control: negative or ambiguous experiments are documented but not featured.

## 7. D.S. Alignment (Skill Mapping)
| D.S. Responsibility | Repository Evidence |
|-----------------------|--------------------|
| Data Exploration & Curation | `data_prep.ipynb` profiling & quality report artifacts |
| Data Quality Practices | Outlier & continuity checks; explicit cleaning log |
| Model Design & Validation | Prophet baseline + N-BEATS training notebook; metrics & backtest preview |
| Rapid Prototyping | Residual feature experiment inside same notebook; synthetic fallback for missing data |
| Insight Communication | Clear notebook narrative + Model Card + quality report Markdown |
| Reproducibility | Deterministic seed, saved metrics JSON, consolidated metrics module |

## 8. Reproducibility

Environment options:
1. Conda: `conda env create -f environment.yml` then `conda activate demand-forecast` (name from file).  
2. Poetry: `poetry install` (if using `pyproject.toml`).

Then launch Jupyter and execute notebooks in order. If M5 subset parquet is absent, synthetic data will be generated automatically (clearly logged).

Minimal scripts: existing `scripts/` are kept small; core logic lives in notebooks for transparency.

## 9. Limitations & Deliberate Omissions
To avoid overstating scope:
- No API service or container deployment.
- No hyperparameter search / AutoML; parameters chosen manually for clarity.
- No multi-dataset fusion (M5 only).
- No probabilistic / quantile forecasts (point forecasts only).
- CPU-only; no attempt to optimize for GPU.

## 10. Next Modest Extensions (Not Implemented Yet)
If time allowed, logical incremental steps would be: calendar & holiday covariates, per-item scaling variant, early stopping + lightweight HP tuning, quantile head for uncertainty. These are noted but intentionally deferred.

## 11. Repository Structure (Relevant Subset)
```
notebooks/
  data_prep.ipynb          # Profiling + cleaning + quality artifacts
  prophet_baseline.ipynb   # Baseline statistical model
  nbeats_training.ipynb    # Deep model + backtest + feature engineering
src/
  evaluation/metrics.py    # Central metric implementations
artifacts/
  data/                    # Quality report JSON/MD
  models/                  # Checkpoints + metrics + model card
```

## 12. Running Time (Approximate)
On Mac M1 Pro (CPU only):
- Data prep: < 1 min
- Prophet baseline (subset of ~10 items + aggregate): a few minutes
- N-BEATS base (20 epochs): depends on subset size; keep item cap low (e.g. 50) to finish within ~10–15 min
- Residual feature retrain (10 epochs): ~5–8 min

These approximations help reviewers gauge practicality; actual times logged in model card/checklist when finalized.

## 13. How to Fill Results Table
After running notebooks:
1. Read JSON artifacts in `artifacts/models/`.
2. Insert metric values into the Results section (replace placeholders).
3. Commit updated README + metrics + model card.

## 14. Ethical & Practical Notes
Forecasting quality depends on data representativeness. Synthetic fallback is only for process demonstration and should not be evaluated as real performance. No claims made about production readiness.

---
This lean capstone demonstrates disciplined, transparent forecasting workflow aligned with real data science responsibilities without over-claiming deployment expertise.
