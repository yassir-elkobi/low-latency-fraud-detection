from typing import Any, Dict, List
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from fastapi import APIRouter, HTTPException
from ..state import AppState
from ..schemas import MetricsOut

"""
Service metrics endpoint router skeleton.

Provides a GET /metrics endpoint exposing request count, tail latency
percentiles (p50/p95/p99), and a basic RPS estimate for the dashboard.
"""


class ServiceMetricsRouter:
    """Router encapsulating service-level metrics endpoints."""

    def __init__(self, state: AppState) -> None:
        self.state = state
        self.router = APIRouter()
        self.router.add_api_route("/metrics", self.get_metrics, methods=["GET"], response_model=MetricsOut)
        self.router.add_api_route("/metrics/offline", self.get_offline_metrics, methods=["GET"])
        self.router.add_api_route("/metrics/stream", self.get_stream_metrics, methods=["GET"])
        self.router.add_api_route("/metrics/stream/ablate", self.run_ablation, methods=["POST"])
        self.router.add_api_route("/metrics/ops", self.get_ops, methods=["GET"])

    def get_metrics(self) -> MetricsOut:
        """Return latency percentiles and request counters for observability."""
        buf = self.state.get_latency_buffer()
        count = buf.count()
        # Use 5-minute window for percentiles to align tails
        WINDOW = 300.0
        p = buf.percentiles_in_window([50, 95, 99], WINDOW)
        # Sliding-window RPS views
        rps_30s = buf.rps(window_seconds=30.0)
        rps_5m = buf.rps(window_seconds=WINDOW)
        count_30s = buf.count_in_window(30.0)
        count_5m = buf.count_in_window(WINDOW)
        return MetricsOut(
            count=int(count),
            count_30s=int(count_30s),
            count_5m=int(count_5m),
            p50_ms=float(p.get(50, 0.0)),
            p95_ms=float(p.get(95, 0.0)),
            p99_ms=float(p.get(99, 0.0)),
            rps=rps_30s,
            rps_5m=rps_5m,
        )

    def get_ops(self) -> Dict[str, float]:
        """Return background worker operational counters."""
        return self.state.get_ops()

    def get_offline_metrics(self) -> Dict[str, Any]:
        """Serve offline metrics JSON produced by training/evaluation."""
        path = Path("models/metrics_offline.json")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Offline metrics not found. Run training.")
        return json.loads(path.read_text())

    def get_stream_metrics(self, limit: int = 500) -> Dict[str, List[float]]:
        """Serve recent streaming coverage points from artifacts CSV.

        Returns last `limit` points of idx, coverage, and violations arrays.
        """
        path = Path("artifacts/streaming_metrics.csv")
        if not path.exists():
            raise HTTPException(status_code=404, detail="Streaming metrics not found. Run streaming simulation.")
        df = pd.read_csv(path)
        if limit and limit > 0:
            df = df.tail(limit)
        out: Dict[str, List[float]] = {
            "idx": df["idx"].astype(float).tolist(),
            "coverage": df["coverage"].astype(float).tolist(),
            "violations": df["violations"].astype(float).tolist(),
        }
        if "coverage_pos" in df.columns:
            out["coverage_pos"] = df["coverage_pos"].astype(float).tolist()
        if "coverage_neg" in df.columns:
            out["coverage_neg"] = df["coverage_neg"].astype(float).tolist()
        return out

    def run_ablation(self) -> Dict[str, Any]:
        """Kick off streaming ablation in a background process.

        Runs `python -m scripts.simulate_stream --ablate` and returns immediately.
        """
        cmd = [sys.executable, "-m", "scripts.simulate_stream", "--ablate"]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to start ablation: {exc}")
        return {"status": "started"}
