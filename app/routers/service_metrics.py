from typing import Any, Dict


class ServiceMetricsRouter:
    """Router for service performance metrics such as tail latencies and RPS.

    Provides endpoints that summarize recent request latencies using a ring
    buffer and expose percentiles (p50/p95/p99) and request rate estimates.
    """

    def register(self) -> None:
        """Register the /metrics route on the FastAPI application/router."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Return current service metrics for dashboard consumption."""
        pass


"""
Service metrics endpoint router skeleton.

Provides a GET /metrics endpoint exposing request count, tail latency
percentiles (p50/p95/p99), and a basic RPS estimate for the dashboard.
"""

from ..schemas import MetricsOut


class ServiceMetricsRouter:
    """Router encapsulating service-level metrics endpoints."""

    def get_metrics(self) -> MetricsOut:
        """Return latency percentiles and request counters for observability."""
        pass
