from typing import Optional, Any
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from .state import AppState
from .routers.health import HealthRouter
from .routers.predict import PredictRouter
from .routers.service_metrics import ServiceMetricsRouter
from .routers.admin import AdminRouter

DEFAULT_TITLE = "Low Latency Fraud Detection"
DEFAULT_VERSION = "1.0.0"
DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_RING_BUFFER_SIZE = 20000
DEFAULT_STATIC_DIR = "app/static"
DEFAULT_TEMPLATES_DIR = "app/templates"

"""FastAPI application factory and bootstrapper for LowLatencyFraudDetection.

Responsible for constructing the FastAPI app, attaching middleware for
latency instrumentation, mounting static files and Jinja templates, and
registering API routers for health, prediction, and service metrics.
"""


class Application:
    """Builder for the FastAPI app with middleware, routers, and dashboard."""

    def __init__(
            self,
            title: str = DEFAULT_TITLE,
            version: str = DEFAULT_VERSION,
            model_path: str = DEFAULT_MODEL_PATH,
            ring_buffer_size: int = DEFAULT_RING_BUFFER_SIZE,
    ) -> None:
        self.title = title
        self.version = version
        self.model_path = model_path
        self.ring_buffer_size = ring_buffer_size
        self.app: Optional[FastAPI] = None
        self.state: Optional[AppState] = None
        self.templates: Optional[Jinja2Templates] = None

    def build(self) -> "Application":
        """Construct the FastAPI app instance and wire core middleware."""
        self.app = FastAPI(title=self.title, version=self.version)
        self.state = AppState(model_path=self.model_path, ring_buffer_size=self.ring_buffer_size)
        self._add_latency_middleware()
        assert self.app is not None and self.state is not None
        self.app.add_event_handler("shutdown", lambda: self.state.shutdown_background())
        return self

    def include_routers(self) -> "Application":
        """Register API routers for health, prediction, and metrics."""
        assert self.app is not None and self.state is not None
        health = HealthRouter(self.state)
        predict = PredictRouter(self.state)
        metrics = ServiceMetricsRouter(self.state)
        admin = AdminRouter(self.state)
        self.app.include_router(health.router)
        self.app.include_router(predict.router)
        self.app.include_router(metrics.router)
        self.app.include_router(admin.router)
        return self

    def mount_static(self, static_path: Optional[str] = None) -> "Application":
        """Mount the static file directory for dashboard assets."""
        assert self.app is not None
        static_dir = static_path or DEFAULT_STATIC_DIR
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        self.app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")
        return self

    def mount_templates(self, templates_path: Optional[str] = None) -> "Application":
        """Mount the Jinja2 templates directory for the dashboard."""
        assert self.app is not None
        templates_dir = templates_path or DEFAULT_TEMPLATES_DIR
        self.templates = Jinja2Templates(directory=templates_dir)
        self._register_dashboard_route()
        return self

    def _add_latency_middleware(self) -> None:
        assert self.app is not None and self.state is not None

        @self.app.middleware("http")
        async def latency_middleware(request: Request, call_next):  # type: ignore[override]
            t0_ns = time.perf_counter_ns()
            response = await call_next(request)
            dt_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
            try:
                # Record only /predict POST requests to avoid polluting with health/metrics
                if request.method == "POST" and request.url.path == "/predict":
                    self.state.get_latency_buffer().append(dt_ms)  # type: ignore[union-attr]
                    # Debounced live-metrics flush in background (no hot-path I/O)
                    self.state.schedule_metrics_flush()  # type: ignore[union-attr]
            except Exception:
                # Never let metrics recording break requests
                pass
            return response

    def _register_dashboard_route(self) -> None:
        assert self.app is not None and self.templates is not None

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request) -> HTMLResponse:  # type: ignore[override]
            return self.templates.TemplateResponse("index.html", {"request": request})


def create_app() -> Any:
    """Create and configure the FastAPI application instance."""
    app = Application().build().include_routers().mount_static().mount_templates()
    return app.app  # type: ignore[return-value]
