from typing import Any, Dict
from fastapi import APIRouter
from ..state import AppState
from ..schemas import HealthOut

"""
Health endpoint router skeleton.

Provides a GET /health endpoint to report service status and basic
model metadata for monitoring and readiness checks.
"""


class HealthRouter:
    """Router encapsulating health-check related endpoints."""

    def __init__(self, state: AppState) -> None:
        self.state = state
        self.router = APIRouter()
        self.router.add_api_route("/health", self.get_health, methods=["GET"], response_model=HealthOut)

    def get_health(self) -> HealthOut:
        """Return service health status and minimal model metadata."""
        info = self.state.get_model_info()
        return HealthOut(status="ok", model_loaded=bool(info.get("loaded", False)), model_info=info)
