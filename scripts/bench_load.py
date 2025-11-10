from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from queue import Queue, Empty
from typing import Any, Dict, List

import requests


def worker(url: str, payload: Dict[str, Any], out_q: Queue, stop_at: float) -> None:
    session = requests.Session()
    while time.time() < stop_at:
        t0 = time.perf_counter_ns()
        try:
            r = session.post(url, json=payload, timeout=10)
            r.raise_for_status()
        except Exception:
            continue
        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        out_q.put(dt_ms)


def run_load(host: str, port: int, concurrency: int, duration: float, example: Dict[str, Any]) -> Dict[str, float]:
    url = f"http://{host}:{port}/predict"
    payload = example
    stop_at = time.time() + duration
    out_q: Queue = Queue()
    threads: List[threading.Thread] = []
    for _ in range(concurrency):
        th = threading.Thread(target=worker, args=(url, payload, out_q, stop_at), daemon=True)
        th.start()
        threads.append(th)
    latencies: List[float] = []
    while any(th.is_alive() for th in threads) or not out_q.empty():
        try:
            val = out_q.get(timeout=0.2)
            latencies.append(float(val))
        except Empty:
            pass
        if time.time() >= stop_at and all(not th.is_alive() for th in threads):
            break
    for th in threads:
        th.join(timeout=0.5)
    if not latencies:
        return {"count": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "rps": 0.0}
    latencies_sorted = sorted(latencies)
    def pct(p: float) -> float:
        if not latencies_sorted:
            return 0.0
        k = (p / 100.0) * (len(latencies_sorted) - 1)
        lo = int(k)
        hi = min(lo + 1, len(latencies_sorted) - 1)
        frac = k - lo
        return latencies_sorted[lo] * (1 - frac) + latencies_sorted[hi] * frac
    rps = len(latencies) / duration
    return {
        "count": float(len(latencies)),
        "p50_ms": pct(50.0),
        "p95_ms": pct(95.0),
        "p99_ms": pct(99.0),
        "mean_ms": statistics.fmean(latencies),
        "rps": rps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple /predict load tester")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--example", type=str, default="", help="JSON string of predict payload")
    args = parser.parse_args()
    if args.example:
        try:
            payload = json.loads(args.example)
        except Exception as exc:
            raise SystemExit(f"Invalid --example JSON: {exc}")
    else:
        # Fallback: minimal payload expecting features list of zeros
        payload = {"features": [0] * 30}
    result = run_load(args.host, args.port, args.concurrency, args.duration, payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
