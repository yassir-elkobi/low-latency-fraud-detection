from typing import Any, Dict


class PredictRouter:
    """Router for prediction endpoints serving calibrated probabilities.

    Exposes a POST endpoint that accepts feature vectors or mappings and
    returns the predicted probability, hard label, and measured latency.
    """

    def register(self) -> None:
        """Register the /predict route on the FastAPI application/router."""
        pass

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input payload and return the prediction response schema."""
        pass


"""
Prediction endpoint router skeleton.

Provides a POST /predict endpoint to score incoming feature vectors with
the calibrated model and return probability, hard label, and latency.
"""

from ..schemas import PredictIn, PredictOut


class PredictRouter:
    """Router encapsulating prediction-related endpoints."""

    def predict(self, payload: PredictIn) -> PredictOut:
        """Score the input features and return probability, label, and latency."""
        pass
