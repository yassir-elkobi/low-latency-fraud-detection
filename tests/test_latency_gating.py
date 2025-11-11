from fastapi.testclient import TestClient
from joblib import load
from app.main import create_app


def _payload_from_model():
    model = load("models/model.joblib")
    underlying = getattr(model, "estimator", None) or getattr(model, "base_estimator", None) or model
    pre = underlying.named_steps.get("pre")
    cols = list(pre.transformers_[0][2])
    return {"features": {c: 0.0 for c in cols}}


def test_latency_gating_and_order() -> None:
    client = TestClient(create_app())
    payload = _payload_from_model()
    n = 600
    for _ in range(n):
        client.post("/predict", json=payload)
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["count"] >= n
    # Order sanity (allow equality in degenerate cases, but ensure some spread)
    assert m["p50_ms"] <= m["p95_ms"] <= m["p99_ms"]
    assert m["p99_ms"] > m["p50_ms"]
