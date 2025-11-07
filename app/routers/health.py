from typing import Any, Dict


class HealthRouter:
    """Router for service liveness/readiness and model health endpoints.

    Provides a lightweight endpoint to report service status, whether the
    model is loaded, and minimal model metadata for operators and tests.
    """

    def register(self) -> None:
        """Register the /health route on the FastAPI application/router."""
        pass

    def get_health(self) -> Dict[str, Any]:
        """Return current health status and model information."""
        pass


"""
Health endpoint router skeleton.

Provides a GET /health endpoint to report service status and basic
model metadata for monitoring and readiness checks.
"""

from typing import Any

from ..schemas import HealthOut


class HealthRouter:
    """Router encapsulating health-check related endpoints."""

    def get_health(self) -> HealthOut:
        """Return service health status and minimal model metadata."""
        pass
