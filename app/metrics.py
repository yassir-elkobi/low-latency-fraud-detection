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
        pass

    def append(self, value_ms: float) -> None:
        """Append a new latency value in milliseconds into the ring buffer."""
        pass

    def snapshot(self) -> List[float]:
        """Return a snapshot copy of the current latency values."""
        pass

    def count(self) -> int:
        """Return the current number of stored latency values."""
        pass

    def percentiles(self, qs: List[float]) -> Dict[float, float]:
        """Compute requested percentiles (0–100) over the stored latencies."""
        pass
