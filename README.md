# Low-Latency Fraud Detection

Real-time fraud detection microservice with calibrated ML models, tail-latency monitoring (P50/P95/P99), and streaming
evaluation under drift. Built for production-grade decisioning with conformal prediction guarantees.

## Quickstart

1. python3 -m venv .venv && source .venv/bin/activate
2. make install
3. python -m scripts.train_baseline
4. (optional) make stream
5. make serve # http://localhost:8000 (dashboard at /)

## Layout

- app/: FastAPI app, routers, schemas, state, metrics, dashboard assets
- scripts/: training, offline eval, streaming simulator, shared utilities
- tests/: API, latency, calibration tests
- data/, models/, artifacts/: inputs and outputs (git-kept via .gitkeep)

## Data splits (offline)

- Strict three-way split:
  - Train/Valid: select base model on a held-out valid split (e.g., AP); then refit base on train+valid.
  - Calibrate: fit `CalibratedClassifierCV(..., cv="prefit")` on the calibration split only (no leakage).
  - Test: compute ROC-AUC, PR-AUC, Brier, and NLL on the test split only.
  - Baselines (CV): the dashboard’s CV table is 5-fold Stratified CV run on train+valid only; the test split remains untouched for final reporting.

Example (values produced by CI; see `models/metrics_offline.json` and the dashboard):

| Metric | Pre (test) | Post (test) |
|-------:|-----------:|------------:|
| ROC-AUC | ~0.98 | ~0.976 |
| PR-AUC  | ~0.746 | ~0.732 |
| Brier   | ~0.023 | ~0.0005 |
| NLL     | ~0.100 | ~0.0039 |

## Metrics (report)

- Offline: ROC-AUC, PR-AUC, Brier, NLL; reliability diagrams pre/post calibration
- Note: with extreme class imbalance, calibrated scores typically live in [0, 0.3]; plots are annotated accordingly.
- Serving: P50/P95/P99 tail latencies, requests/sec (dashboard)
- Streaming: coverage (1-α) and violations vs time; window/decay ablation

## Academic Context

Part of RCP209 (Machine Learning) coursework at CNAM Paris, focused on production ML systems.

- Dataset: Credit Card Fraud Detection (ULB/Kaggle)
- Report: English, covering calibration, conformal prediction, and streaming evaluation
- Key differentiators: P95 latency budgets, coverage guarantees under drift

## API

- GET `/health`: status + model metadata
- POST `/predict`: `{features: [... or {name:value}]}` → `{proba,label,latency_ms}`
- GET `/metrics`: `{count,p50_ms,p95_ms,p99_ms,rps}`

## CI

- Workflow runs: training → offline evaluation → streaming simulation (bounded) → tests
- The live dashboard is served by FastAPI locally

## Deploy (Fly.io)

Prereqs:

- Create a Fly.io app (`low-latency-fraud-detection`), generate an App Deploy Token, add it as repo secret:
  `FLY_API_TOKEN`.

Deploy on push to main (CI will):

- train the model (produces `models/model.joblib`),
- build Docker image and `flyctl deploy`,
- expose at `https://low-latency-fraud-detection.fly.dev`.
