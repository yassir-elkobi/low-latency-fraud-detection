from threading import Lock
from typing import Any, Dict, Optional

from .metrics import LatencyRingBuffer


class AppState:
    """Centralized application state for the StreamScore service.

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
        pass

    def load_model(self, model_path: str) -> None:
        """Load the calibrated model artifact from the given path."""
        pass

    def get_model(self) -> Any:
        """Return the currently loaded model instance (if any)."""
        pass

    def get_latency_buffer(self) -> LatencyRingBuffer:
        """Return the shared latency ring buffer instance."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Return minimal model metadata for health and dashboard endpoints."""
        pass
