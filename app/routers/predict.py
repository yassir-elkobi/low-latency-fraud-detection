from typing import Any, List
import time
import pandas as pd
from fastapi import APIRouter, HTTPException
from ..state import AppState
from ..schemas import PredictIn, PredictOut

"""
Prediction endpoint router skeleton.

Provides a POST /predict endpoint to score incoming feature vectors with
the calibrated model and return probability, hard label, and latency.
"""


class PredictRouter:
    """Router encapsulating prediction-related endpoints."""

    def __init__(self, state: AppState) -> None:
        self.state = state
        self.router = APIRouter()
        self.router.add_api_route("/predict", self.predict, methods=["POST"], response_model=PredictOut)
        self.router.add_api_route("/predict/schema", self.schema, methods=["GET"])

    def _expected_columns(self, model: Any) -> List[str]:
        """Infer expected feature column order from the model's preprocessing pipeline."""
        # Resolve underlying Pipeline regardless of CalibratedClassifierCV wrapper
        try:
            underlying = None
            if hasattr(model, "estimator"):
                underlying = getattr(model, "estimator")
            elif hasattr(model, "base_estimator"):
                underlying = getattr(model, "base_estimator")
            else:
                underlying = model
            pre = underlying.named_steps.get("pre")  # type: ignore[attr-defined]
            cols = pre.transformers_[0][2]  # type: ignore[index]
            return list(cols)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model preprocessor columns could not be determined: {exc}")

    def predict(self, payload: PredictIn) -> PredictOut:
        """Score the input features and return probability, label, and latency."""
        model = self.state.get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        features = payload.features
        expected_cols = self._expected_columns(model)

        if isinstance(features, list):
            if len(features) != len(expected_cols):
                raise HTTPException(status_code=400,
                                    detail=f"Expected {len(expected_cols)} features, got {len(features)}")
            df = pd.DataFrame([features], columns=expected_cols)
        else:
            missing = [c for c in expected_cols if c not in features]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
            # Order columns
            row = [features[c] for c in expected_cols]
            df = pd.DataFrame([row], columns=expected_cols)

        t0 = time.perf_counter()
        proba = float(model.predict_proba(df)[:, 1][0])
        label = int(proba >= 0.5)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictOut(proba=proba, label=label, latency_ms=latency_ms)

    def schema(self) -> Any:
        """Return expected feature columns and an example payload."""
        model = self.state.get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        cols = self._expected_columns(model)
        example = {"features": {c: 0 for c in cols}}
        return {"expected_columns": cols, "example": example}
