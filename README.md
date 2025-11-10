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

- Train/Valid: select base model on a held-out valid split (e.g., AP); then refit base on train+valid.
- Calibrate: fit `CalibratedClassifierCV(..., cv="prefit")` on the calibration split only.
- Test: report ROC-AUC, PR-AUC, Brier, and NLL on the test split only.

## Metrics (report)

- Offline: ROC-AUC, PR-AUC, Brier, NLL; reliability diagrams pre/post calibration
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
