from collections import deque
from threading import Lock
from typing import Dict, List, Optional


class LatencyRingBuffer:
    """Thread-safe fixed-size ring buffer for request latencies (milliseconds).

    Stores recent latency measurements up to a maximum capacity (maxlen) and
    exposes utilities to compute tail latency percentiles (p50/p95/p99) on
    demand for observability and dashboarding.
    """

    _maxlen: int
    _buffer: Optional[deque] = None
    _lock: Optional[Lock] = None

    def __init__(self, maxlen: int = 20000) -> None:
        """Initialize a ring buffer with a maximum number of stored entries."""
        self._maxlen = maxlen
        self._buffer = deque(maxlen=maxlen)
        self._lock = Lock()

    def append(self, value_ms: float) -> None:
        """Append a new latency value in milliseconds into the ring buffer."""
        if value_ms < 0:
            return
        with self._lock:  # type: ignore[arg-type]
            self._buffer.append(float(value_ms))  # type: ignore[union-attr]

    def snapshot(self) -> List[float]:
        """Return a snapshot copy of the current latency values."""
        with self._lock:  # type: ignore[arg-type]
            return list(self._buffer)  # type: ignore[union-attr]

    def count(self) -> int:
        """Return the current number of stored latency values."""
        with self._lock:  # type: ignore[arg-type]
            return len(self._buffer)  # type: ignore[union-attr]

    def percentiles(self, qs: List[float]) -> Dict[float, float]:
        """Compute requested percentiles (0–100) over the stored latencies."""
        values = self.snapshot()
        if not values:
            return {q: 0.0 for q in qs}
        arr = sorted(values)
        out: Dict[float, float] = {}
        for q in qs:
            if q <= 0:
                out[q] = arr[0]
                continue
            if q >= 100:
                out[q] = arr[-1]
                continue
            k = (q / 100.0) * (len(arr) - 1)
            lo = int(k)
            hi = min(lo + 1, len(arr) - 1)
            frac = k - lo
            out[q] = arr[lo] * (1 - frac) + arr[hi] * frac
        return out
