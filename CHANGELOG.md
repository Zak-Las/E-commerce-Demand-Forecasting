# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-04
### Added
- Baseline evaluation module (`src/evaluation/baselines.py`) with Naive Last Value and Seasonal Weekly forecasts.
- Unified metric suite (RMSE, MAE, MAPE, sMAPE, WAPE, Forecast Accuracy).
- Artifact generation: per-item parquet metrics + JSON summary.
- Initial test suite (31 tests): metrics, edge cases, parametric validation, API smoke tests.
- WAPE and forecast accuracy integrated across evaluations.
- README updated with WAPE formula and real benchmark comparison table.
- `pyproject.toml` and version constant (`__version__ = "0.1.0"`).
- `CHANGELOG.md` following Keep a Changelog format.

### Notes
This is the first tagged foundation release establishing reliable baselines prior to deep learning models (N-BEATS, TFT) and probabilistic extensions.

## [Unreleased]
### Planned
- N-BEATS and TFT model training pipelines.
- Rolling backtest orchestration CLI.
- Probabilistic metrics (pinball loss, CRPS).
- Forecast API endpoint integration with real model outputs.
- CI workflow (pytest, style checks, artifact validation).
- Model version registry and comparison dashboard.

[0.1.0]: https://github.com/Zak-Las/E-commerce-Demand-Forecasting/releases/tag/v0.1.0