from typing import Optional


class Application:
    """FastAPI application factory and bootstrapper for StreamScore.

    Responsible for constructing the FastAPI app, attaching middleware for
    latency instrumentation, mounting static files and Jinja templates, and
    registering API routers for health, prediction, and service metrics.
    """

    def __init__(self, title: str = "StreamScore", version: str = "0.1.0") -> None:
        """Initialize the application metadata and internal placeholders."""
        pass

    def build(self) -> "Application":
        """Construct the FastAPI app instance and wire core middleware."""
        pass

    def include_routers(self) -> "Application":
        """Register API routers for health, prediction, and metrics."""
        pass

    def mount_static(self, static_path: Optional[str] = None) -> "Application":
        """Mount the static file directory for dashboard assets."""
        pass

    def mount_templates(self, templates_path: Optional[str] = None) -> "Application":
        """Mount the Jinja2 templates directory for the dashboard."""
        pass


"""
FastAPI application factory and integration points.

Exposes a function to create and configure the FastAPI app, including
middleware for latency instrumentation, routers registration, and
templating/static mounting for the minimal dashboard.
"""

from typing import Any


def create_app() -> Any:
    """Create and configure the FastAPI application instance."""
    pass
