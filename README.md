# RCP209 – StreamScore

FastAPI microservice serving a calibrated binary classifier with tail-latency (P50/P95/P99) metrics and a minimal
dashboard, plus offline evaluation and a streaming simulator (coverage/violations under drift and label delay).

## Quickstart

1. python3 -m venv .venv && source .venv/bin/activate
2. make install
3. python scripts/train_baseline.py
4. (optional) make stream
5. (later) serve via uvicorn once app factory is implemented

## Layout

- app/: FastAPI app, routers, schemas, state, metrics, dashboard assets
- scripts/: training, offline eval, streaming simulator, data download
- tests/: API, latency, calibration tests (skeletons)
- data/, models/, artifacts/: inputs and outputs (git-kept via .gitkeep)

## Metrics (report)

- Offline: ROC-AUC, PR-AUC, Brier, NLL; reliability diagrams pre/post calibration
- Serving: P50/P95/P99 tail latencies, requests/sec (dashboard)
- Streaming: coverage (1-α) and violations vs time; window/decay ablation

## Notes

- Dataset: Credit Card Fraud (ULB/Kaggle). Provide data/creditcard.csv locally or use a sample.
- Report language: English, aligned with RCP209 expectations.
