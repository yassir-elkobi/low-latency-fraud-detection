from collections import deque
from threading import Lock
from typing import Dict, List, Optional
import time


class LatencyRingBuffer:
    """Thread-safe fixed-size ring buffer for request latencies (milliseconds).

    Stores recent latency measurements up to a maximum capacity (maxlen) and
    exposes utilities to compute tail latency percentiles (p50/p95/p99) on
    demand for observability and dashboarding.
    """

    _maxlen: int
    _buffer: Optional[deque] = None
    _lock: Optional[Lock] = None
    _timebuffer: Optional[deque] = None

    def __init__(self, maxlen: int = 20000) -> None:
        """Initialize a ring buffer with a maximum number of stored entries."""
        self._maxlen = maxlen
        self._buffer = deque(maxlen=maxlen)
        self._lock = Lock()
        self._timebuffer = deque(maxlen=maxlen)

    def append(self, value_ms: float) -> None:
        """Append a new latency value in milliseconds into the ring buffer."""
        if value_ms < 0:
            return
        now = time.time()
        with self._lock:  # type: ignore[arg-type]
            self._buffer.append(float(value_ms))  # type: ignore[union-attr]
            self._timebuffer.append(now)  # type: ignore[union-attr]

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

    def rps(self, window_seconds: float = 30.0) -> float:
        """Compute requests-per-second over the last window_seconds."""
        if window_seconds <= 0:
            return 0.0
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:  # type: ignore[arg-type]
            times = list(self._timebuffer)  # type: ignore[union-attr]
        if not times:
            return 0.0
        # Count timestamps within the window
        count = 0
        for t in reversed(times):
            if t >= cutoff:
                count += 1
            else:
                break
        return float(count) / window_seconds

    def count_in_window(self, window_seconds: float) -> int:
        """Count number of entries within the last window_seconds."""
        if window_seconds <= 0:
            return 0
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:  # type: ignore[arg-type]
            times = list(self._timebuffer)  # type: ignore[union-attr]
        c = 0
        for t in reversed(times):
            if t >= cutoff:
                c += 1
            else:
                break
        return c

    def percentiles_in_window(self, qs: List[float], window_seconds: float) -> Dict[float, float]:
        """Compute percentiles over entries within the last window_seconds."""
        if window_seconds <= 0:
            return {q: 0.0 for q in qs}
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:  # type: ignore[arg-type]
            times = list(self._timebuffer)  # type: ignore[union-attr]
            values = list(self._buffer)  # type: ignore[union-attr]
        if not times or not values:
            return {q: 0.0 for q in qs}
        win_vals: List[float] = []
        for idx in range(len(times) - 1, -1, -1):
            if times[idx] >= cutoff:
                win_vals.append(values[idx])
            else:
                break
        if not win_vals:
            return {q: 0.0 for q in qs}
        arr = sorted(win_vals)
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
