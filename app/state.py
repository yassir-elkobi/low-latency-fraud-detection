from threading import Lock, Thread
from queue import Queue, Empty
from typing import Any, Callable, Dict, Optional, Tuple
from .metrics import LatencyRingBuffer
from pathlib import Path
from joblib import load
import time
import json
import os


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
    _bg_thread: Optional[Thread] = None
    _bg_queue: Optional[Queue] = None
    _debounce_registry: Dict[str, Tuple[float, float, Callable[[], None]]] = {}

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
        # Lightweight background worker for off-path tasks (file writes, etc.)
        self._bg_queue = Queue()
        self._bg_thread = Thread(target=self._bg_loop, name="bg-worker", daemon=True)
        self._bg_thread.start()
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load the calibrated model artifact from the given path."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model artifact not found at {p}")
        # Memory-map the model file to reduce RSS and avoid OOM on small VMs
        model = load(p, mmap_mode="r")
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

    # --------------------- background worker utilities ---------------------
    def _bg_loop(self) -> None:
        """Simple background loop processing queued callables and debounced jobs."""
        while True:
            try:
                func = self._bg_queue.get(timeout=0.25)  # type: ignore[arg-type]
                if func is None:  # shutdown signal
                    break
                try:
                    func()
                except Exception:
                    # Background tasks must never crash the process
                    pass
            except Empty:
                pass
            # Check debounced tasks
            now = time.time()
            for name, (last_run, min_interval, fn) in list(self._debounce_registry.items()):
                if now - last_run >= min_interval:
                    # Update last_run before running to avoid tight loops on failure
                    self._debounce_registry[name] = (now, min_interval, fn)
                    try:
                        fn()
                    except Exception:
                        pass

    def submit_background(self, fn: Callable[[], None]) -> None:
        """Submit a callable to be executed in the background."""
        if self._bg_queue is not None:
            self._bg_queue.put(fn)

    def debounce_background(self, name: str, min_interval_seconds: float, fn: Callable[[], None]) -> None:
        """Register or update a debounced task executed no more than once per interval."""
        prev = self._debounce_registry.get(name)
        now = time.time()
        if prev is None:
            self._debounce_registry[name] = (0.0, float(min_interval_seconds), fn)
        else:
            # Update function and interval if they changed
            self._debounce_registry[name] = (prev[0], float(min_interval_seconds), fn)

    def shutdown_background(self) -> None:
        """Signal background worker to stop."""
        if self._bg_queue is not None:
            self._bg_queue.put(None)  # type: ignore[arg-type]
        if self._bg_thread is not None:
            try:
                self._bg_thread.join(timeout=2.0)
            except Exception:
                pass

    # --------------------- debounced metrics flush ---------------------
    def flush_live_metrics(self) -> None:
        """Write a small JSON snapshot of live metrics to artifacts/."""
        buf = self.get_latency_buffer()
        # Compute on 5m window to match UI
        WINDOW = 300.0
        counts = {
            "count_total": buf.count(),
            "count_30s": buf.count_in_window(30.0),
            "count_5m": buf.count_in_window(WINDOW),
        }
        pct = buf.percentiles_in_window([50, 95, 99], WINDOW)
        rps = {
            "rps_30s": buf.rps(30.0),
            "rps_5m": buf.rps(WINDOW),
        }
        payload = {
            "timestamp": time.time(),
            "counts": counts,
            "percentiles_ms": {"p50": pct.get(50, 0.0), "p95": pct.get(95, 0.0), "p99": pct.get(99, 0.0)},
            "rps": rps,
        }
        out_dir = Path("artifacts")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "live_metrics.json.tmp"
            dst = out_dir / "live_metrics.json"
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, dst)
        except Exception:
            # Ignore I/O errors; background flushes are best-effort
            pass

    def schedule_metrics_flush(self, min_interval_seconds: float = 15.0) -> None:
        """Debounce live metrics flush to avoid I/O on the hot path."""
        self.debounce_background("flush_live_metrics", min_interval_seconds, self.flush_live_metrics)
