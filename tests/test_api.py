from typing import List
from fastapi.testclient import TestClient
from joblib import load
from app.main import create_app


def _expected_columns_from_model(model) -> List[str]:
    try:
        underlying = getattr(model, "estimator", None) or getattr(model, "base_estimator", None) or model
        pre = underlying.named_steps.get("pre")
        return list(pre.transformers_[0][2])
    except Exception as exc:
        raise AssertionError(f"Could not extract expected columns from model: {exc}")


def test_health() -> None:
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["model_info"], dict)


def test_predict_schema() -> None:
    client = TestClient(create_app())
    model = load("models/model.joblib")
    cols = _expected_columns_from_model(model)
    payload = {"features": {c: 0.0 for c in cols}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["proba"] <= 1.0
    assert body["label"] in (0, 1)
    assert body["latency_ms"] >= 0.0


def test_predict_invalid_input() -> None:
    client = TestClient(create_app())
    # Wrong number of features with list input
    r = client.post("/predict", json={"features": [0.0, 1.0, 2.0]})
    assert r.status_code in (400, 422)
