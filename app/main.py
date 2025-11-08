from typing import Optional
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from .state import AppState
from .routers.health import HealthRouter
from .routers.predict import PredictRouter
from .routers.service_metrics import ServiceMetricsRouter


class Application:
    """FastAPI application factory and bootstrapper for StreamScore.

    Responsible for constructing the FastAPI app, attaching middleware for
    latency instrumentation, mounting static files and Jinja templates, and
    registering API routers for health, prediction, and service metrics.
    """

    def __init__(self, title: str = "StreamScore", version: str = "0.1.0") -> None:
        """Initialize the application metadata and internal placeholders."""
        self.title = title
        self.version = version
        self.app: Optional[FastAPI] = None
        self.state: Optional[AppState] = None
        self.templates: Optional[Jinja2Templates] = None

    def build(self) -> "Application":
        """Construct the FastAPI app instance and wire core middleware."""
        self.app = FastAPI(title=self.title, version=self.version)
        self.state = AppState(model_path="models/model.joblib", ring_buffer_size=20000)

        @self.app.middleware("http")
        async def latency_middleware(request: Request, call_next):  # type: ignore[override]
            t0 = time.perf_counter()
            response = await call_next(request)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            try:
                self.state.get_latency_buffer().append(latency_ms)  # type: ignore[union-attr]
            except Exception:
                pass
            return response

        return self

    def include_routers(self) -> "Application":
        """Register API routers for health, prediction, and metrics."""
        assert self.app is not None and self.state is not None
        health = HealthRouter(self.state)
        predict = PredictRouter(self.state)
        metrics = ServiceMetricsRouter(self.state)
        self.app.include_router(health.router)
        self.app.include_router(predict.router)
        self.app.include_router(metrics.router)
        return self

    def mount_static(self, static_path: Optional[str] = None) -> "Application":
        """Mount the static file directory for dashboard assets."""
        assert self.app is not None
        static_dir = static_path or "app/static"
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        return self

    def mount_templates(self, templates_path: Optional[str] = None) -> "Application":
        """Mount the Jinja2 templates directory for the dashboard."""
        assert self.app is not None
        templates_dir = templates_path or "app/templates"
        self.templates = Jinja2Templates(directory=templates_dir)

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request) -> HTMLResponse:  # type: ignore[override]
            assert self.templates is not None
            return self.templates.TemplateResponse("index.html", {"request": request})

        return self


"""
FastAPI application factory and integration points.

Exposes a function to create and configure the FastAPI app, including
middleware for latency instrumentation, routers registration, and
templating/static mounting for the minimal dashboard.
"""

from typing import Any


def create_app() -> Any:
    """Create and configure the FastAPI application instance."""
    app = Application().build().include_routers().mount_static().mount_templates()
    return app.app  # type: ignore[return-value]
