from typing import Any, Dict
import os
import time
from fastapi import APIRouter, HTTPException, Header
from ..state import AppState


class AdminRouter:
    """Administrative endpoints guarded by a static token and env toggle.

    Provides a POST /admin/flush endpoint to trigger an immediate background
    metrics flush. Rate-limited to avoid abuse.
    """

    def __init__(self, state: AppState) -> None:
        self.state = state
        self.router = APIRouter()
        self._last_manual_flush_ts: float = 0.0
        self._min_interval_seconds: float = float(os.getenv("ADMIN_FLUSH_MIN_INTERVAL", "10"))
        self.router.add_api_route("/admin/flush", self.flush_now, methods=["POST"])

    def _check_enabled(self) -> None:
        enabled = os.getenv("ENABLE_ADMIN", "false").lower() in ("1", "true", "yes")
        if not enabled:
            raise HTTPException(status_code=403, detail="Admin endpoints disabled")

    def _check_token(self, token: str | None) -> None:
        expected = os.getenv("ADMIN_TOKEN", "")
        if not expected or token != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    def flush_now(self, x_admin_token: str | None = Header(default=None)) -> Dict[str, Any]:
        """Enqueue an immediate metrics flush if allowed; returns job info."""
        self._check_enabled()
        self._check_token(x_admin_token)
        now = time.time()
        if now - self._last_manual_flush_ts < self._min_interval_seconds:
            raise HTTPException(status_code=429, detail="Too many requests")
        self._last_manual_flush_ts = now
        # Enqueue the flush job (best-effort; debounced flush remains active too)
        self.state.submit_background(self.state.flush_live_metrics)
        job_id = f"flush-{int(now)}"
        return {"status": "accepted", "job_id": job_id}
