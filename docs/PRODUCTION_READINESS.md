# Production Readiness Checklist

This document defines the criteria for moving the project from experimental to production-grade. It is organized by domain with gating levels (Alpha, Beta, Production).

## Gating Levels Overview
| Stage | Minimum Required | Optional Enhancements |
|-------|------------------|-----------------------|
| Alpha | Baselines, first advanced model, manual backtests, unit tests, simple API health | Parametric tests, initial CI |
| Beta | Advanced model (N-BEATS), rolling backtests, CI (tests + lint), versioned artifacts, structured logs | Drift checks, load tests, containerization |
| Production | Stable API schema, automated retraining/backtest, full CI (security + type + coverage >70%), release automation, fallback strategy | Quantile metrics, model registry UI, tracing |

---
## 1. Data & Ingestion
- Integrity: Validate file sizes / optional SHA256 for raw M5 dataset.
- Schema contract: `item_id`, `date`, `demand` with explicit dtypes; validation (pandera / Great Expectations).
- Missing values: Policy defined (drop / zero / forward fill) documented.
- Outliers: Criteria (e.g., demand spikes > P99) with business rationale.
- Lineage: Deterministic scripts + versioned outputs (e.g. `m5_panel_subset_v{X}.parquet`).
- Governance: License compliance noted; refresh cadence defined.

## 2. Feature Engineering
- Determinism: Same inputs + seed => identical outputs.
- Configurable: Horizon, lag windows, rolling stats via config file.
- Validation: Unit tests for short history, date gaps, zero-demand sequences.
- Performance: Build wall-clock benchmark recorded and monitored.
- Memory: Strategy for large item counts (chunking / filtering / streaming).

## 3. Modeling
- Baselines: Naive + Seasonal (DONE at v0.1.0).
- Advanced: N-BEATS (target Beta), TFT (future).
- Artifacts: Standard format (`artifacts/models/{model}_v{version}.pt` + JSON metadata).
- Hyperparams: Central config; log all values on train start.
- Backtesting: Rolling-origin multi-window harness (target Beta).
- Evaluation: WAPE primary; sMAPE, MAE, RMSE secondary; thresholds established.
- Reproducibility: Single command to rebuild (data→features→train→evaluate).

## 4. Metrics & Monitoring
- KPIs: WAPE, Forecast Accuracy (100-WAPE), sMAPE, MAE, RMSE.
- Future: Pinball loss / quantile coverage.
- Drift: Distribution shift detection (demand level, seasonality) scheduled.
- Alerts: Thresholds (e.g., WAPE > baseline + X%) define incident triggers.

## 5. API / Service
- Contract: Versioned request/response schema (`api_version`).
- Validation: Product ID existence; horizon bounds.
- Error handling: Structured JSON error codes (4xx/5xx distinction).
- Performance: P95 latency target for typical batch (document CPU vs GPU).
- Fallback: Baseline model used if advanced model fails (graceful degrade).
- Observability: `/health`, `/ready`, optional `/metrics` (Prometheus).

## 6. CI/CD
- Pipeline steps: test → lint → security scan → (optional) type checking.
- Version checks: Tag matches `__version__` & `pyproject.toml`.
- Artifact publishing: Model binaries & evaluation reports archived.
- Rollback: Retain previous release artifacts + one-command revert script.

## 7. Versioning & Releases
- Semantic Versioning enforced.
- Changelog updates per release; no silent changes.
- Pre-release flag used until Production criteria met.
- Release notes include metric deltas & upgrade instructions.

## 8. Security & Compliance
- Secrets: No credentials in repo; `.env` or secret manager.
- Dependency hygiene: Regular `pip-audit` or `safety` scan.
- Transport: HTTPS enforced externally.
- Access: Auth/N auth strategy documented if public exposure planned.

## 9. Performance & Scalability
- Load testing: Simulate concurrent forecasts; document breakdown points.
- Horizontal scaling: Stateless inference design (model loaded per process).
- Resource profiling: Peak RAM & CPU usage for training and inference.
- Optimization: Batch inference vs per-request overhead comparison.

## 10. Reliability & Resilience
- Graceful shutdown: In-flight requests complete before termination.
- Retry/backoff: Transient external I/O (e.g., data fetch) robust against failures.
- Circuit breaker: Optional for external dependencies.
- Degradation plan: Use simpler model under heavy load or partial failure.

## 11. Logging & Tracing
- Structured logging (JSON): request_id, model_version, latency, item_count.
- Error classification: Validation vs system vs model.
- Optional tracing: OpenTelemetry integration for multi-service spans.

## 12. Documentation
- Architecture: Data flow diagram, component interactions.
- Model cards: For each model: training data window, metrics, known limitations.
- Runbooks: Incident response (accuracy drop, latency spike, model load failure).
- Upgrade guide: Steps for migrating minor/major versions.

## 13. Testing Strategy
- Unit tests: Metrics, feature transforms, forecast functions.
- Integration tests: API endpoints + end-to-end inference.
- Backtest regression tests: Ensure new version doesn’t degrade WAPE beyond threshold.
- Property tests: Metric invariants (WAPE ≥ 0, RMSE ≥ MAE conditions).
- Smoke tests: Post-deploy minimal forecast check.

## 14. Risk & Failure Modes
- Data drift: Seasonality shifts render historical models stale.
- Demand sparsity: Intermittent zero-demand periods inflate error variability.
- Model staleness: Delayed retraining reduces accuracy resilience.
- Infrastructure mismatch: GPU not available → fallback path.

## 15. Deployment & Infrastructure (Future)
- Containerization: Minimal image with pinned dependencies.
- Config via env vars: MODEL_NAME, DEFAULT_HORIZON, MAX_ITEMS.
- Orchestration: Kubernetes readiness/liveness checks.
- Cost management: Compute resource tracking & trend analysis.

---
## Current Status (v0.1.0)
- Baselines: COMPLETE.
- Advanced models: NOT STARTED.
- Backtesting harness: PARTIAL (Prophet subset only).
- API contract: BASIC skeleton.
- CI/CD: NOT IMPLEMENTED.
- Observability: MINIMAL.
- Security scanning: NOT IMPLEMENTED.

## Immediate Priority Targets
1. N-BEATS training pipeline & artifact serialization.
2. Backtesting CLI (multi-window & horizon).
3. CI setup: pytest + ruff + security scan + version check.
4. Real inference integration in `/forecast`.
5. Model card template for baselines & N-BEATS.

## Acceptance Threshold Examples (Adjustable)
- WAPE improvement vs seasonal naive sustained ≥ 5–10 percentage points.
- Backtest average WAPE <= single-run WAPE + 15% relative.
- P95 API latency < 500 ms for 30-day forecast of 50 items (CPU baseline).
- Test coverage ≥ 70% for non-generated application code pre-Production.
- One-command reproducibility: `make all` or equivalent executes full pipeline.

---
## Production Readiness Gate
All sections Alpha/Beta completed; production gating review ensures: 
- Stable inference contract; version header published.
- Automated daily or weekly retraining/backtest jobs.
- Alerts wired (WAPE threshold breach, latency spike).
- Documented rollback procedure tested.
- Security & dependency scans passing.

---
## Change Management
Each new feature or model version must:
- Update `CHANGELOG.md` under `Unreleased`.
- Include metric comparison vs previous release.
- Justify resource cost changes (if any).

---
## Audit Trail (Planned)
- Release metadata recorded: version, timestamp, model hash, metric summaries.
- Optional signing of model artifacts (hash + signature).

---
## Appendix: Tooling Recommendations
- Validation: pandera / Great Expectations.
- Metrics packaging: custom Python module + JSON export.
- CI: GitHub Actions (pytest, ruff, pip-audit, artifact upload).
- Load test: Locust or k6 with synthetic forecast requests.
- Logging: structlog or Python standard logging with JSON formatter.
- Drift: Evidently AI (optional) or custom statistical tests.

---
**Status:** Living document—update as architecture evolves.
