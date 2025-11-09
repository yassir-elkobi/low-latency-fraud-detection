from threading import Lock
from typing import Any, Dict, Optional
from .metrics import LatencyRingBuffer
from pathlib import Path
from joblib import load
import time


class AppState:
    """Centralized application state for the LowLatencyFraudDetection service.

    Owns the calibrated model artifact, minimal model metadata, and the shared
    latency ring buffer used for tail-latency metrics. Provides simple accessors
    and lifecycle hooks to load/reload the model and expose runtime state.
    """

    _model: Optional[Any] = None
    _model_info: Dict[str, Any] = {}
    _latency_buffer: Optional[LatencyRingBuffer] = None
    _lock: Optional[Lock] = None

    def __init__(self, model_path: Optional[str] = None, ring_buffer_size: int = 20000) -> None:
        """Initialize state; optionally preload model and latency buffer."""
        self._lock = Lock()
        self._latency_buffer = LatencyRingBuffer(maxlen=ring_buffer_size)
        self._model = None
        self._model_info = {
            "loaded": False,
            "path": model_path or "models/model.joblib",
            "loaded_at": None,
        }
        self._start_time = time.time()
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load the calibrated model artifact from the given path."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found at {p}")
        model = load(p)
        with self._lock:  # type: ignore[arg-type]
            self._model = model
            self._model_info.update({
                "loaded": True,
                "path": str(p.as_posix()),
                "loaded_at": time.time(),
                "type": type(model).__name__,
            })

    def get_model(self) -> Any:
        """Return the currently loaded model instance (if any)."""
        return self._model

    def get_latency_buffer(self) -> LatencyRingBuffer:
        """Return the shared latency ring buffer instance."""
        return self._latency_buffer  # type: ignore[return-value]

    def get_model_info(self) -> Dict[str, Any]:
        """Return minimal model metadata for health and dashboard endpoints."""
        return dict(self._model_info)

    def uptime_seconds(self) -> float:
        return max(0.0, time.time() - getattr(self, "_start_time", time.time()))
