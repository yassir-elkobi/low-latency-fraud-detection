from fastapi.testclient import TestClient
from joblib import load
from app.main import create_app


def _features_payload():
    model = load("models/model.joblib")
    underlying = getattr(model, "estimator", None) or getattr(model, "base_estimator", None) or model
    pre = underlying.named_steps.get("pre")
    cols = list(pre.transformers_[0][2])
    return {"features": {c: 0.0 for c in cols}}


def test_latency_percentiles() -> None:
    client = TestClient(create_app())
    payload = _features_payload()
    # Warm-up and generate load
    for _ in range(50):
        client.post("/predict", json=payload)
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["count"] >= 50
    assert m["p50_ms"] <= m["p95_ms"] <= m["p99_ms"]
    assert m["p50_ms"] >= 0.0 and m["p99_ms"] >= 0.0
