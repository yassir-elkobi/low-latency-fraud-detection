from __future__ import annotations

from joblib import load
from scripts.common import load_dataset, temporal_split
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cutoff = q * cw[-1]
    idx = np.searchsorted(cw, cutoff, side="left")
    idx = min(idx, len(v) - 1)
    return float(v[idx])


@dataclass
class WindowBuf:
    maxlen: int
    data: Deque[float]

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)

    def add(self, s: float) -> None:
        self.data.append(s)

    def quantile(self, q: float) -> float:
        if not self.data:
            return float("nan")
        arr = np.asarray(self.data, dtype=float)
        return float(np.quantile(arr, q, method="nearest"))


@dataclass
class ExpBuf:
    decay: float
    values: List[float]
    weights: List[float]

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.values = []
        self.weights = []

    def add(self, s: float) -> None:
        self.weights = [w * (1.0 - self.decay) for w in self.weights]
        self.values.append(float(s))
        self.weights.append(1.0)

    def quantile(self, q: float) -> float:
        if not self.values:
            return float("nan")
        v = np.asarray(self.values, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        return weighted_quantile(v, w, q)


class StreamSimulator:
    """Streaming simulator for online split-conformal with window/decay buffers."""

    def __init__(self, alpha: float = 0.05, mode: str = "window", window: int = 2000, decay: float = 0.01,
                 label_delay: int = 200) -> None:
        self.alpha = alpha
        self.q = 1.0 - alpha
        self.mode = mode
        self.window = window
        self.decay = decay
        self.label_delay = label_delay
        self.buf = WindowBuf(window) if mode == "window" else ExpBuf(decay)

    def simulate(self, probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        pending: Deque[Tuple[int, int]] = deque()
        total = covered = violations = 0
        idx_hist: List[int] = []
        cov_hist: List[float] = []
        viol_hist: List[float] = []

        for i, p1 in enumerate(probs):
            # Predictive set using current tau
            tau = self.buf.quantile(self.q)
            include_one = include_zero = True if np.isnan(tau) else False
            if not np.isnan(tau):
                include_one = p1 >= (1.0 - tau)
                include_zero = (1.0 - p1) >= (1.0 - tau)

            pending.append((i, int(y_true[i])))
            if len(pending) > self.label_delay:
                j, yj = pending.popleft()
                pj = float(probs[j])
                s = pj if yj == 0 else (1.0 - pj)
                self.buf.add(s)
                tau_now = self.buf.quantile(self.q)
                if np.isnan(tau_now):
                    in_set = True
                else:
                    in_set = (yj == 1 and pj >= (1.0 - tau_now)) or (yj == 0 and (1.0 - pj) >= (1.0 - tau_now))
                total += 1
                covered += int(in_set)
                violations += int(not in_set)
                idx_hist.append(total)
                cov_hist.append(covered / total)
                viol_hist.append(violations / total)

        # flush
        while pending:
            j, yj = pending.popleft()
            pj = float(probs[j])
            s = pj if yj == 0 else (1.0 - pj)
            self.buf.add(s)
            tau_now = self.buf.quantile(self.q)
            if np.isnan(tau_now):
                in_set = True
            else:
                in_set = (yj == 1 and pj >= (1.0 - tau_now)) or (yj == 0 and (1.0 - pj) >= (1.0 - tau_now))
            total += 1
            covered += int(in_set)
            violations += int(not in_set)
            idx_hist.append(total)
            cov_hist.append(covered / total)
            viol_hist.append(violations / total)

        return {
            "idx": idx_hist,
            "coverage": cov_hist,
            "violations": viol_hist,
            "final_coverage": cov_hist[-1] if cov_hist else None,
            "final_violation_rate": viol_hist[-1] if viol_hist else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming conformal simulator")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mode", choices=["window", "exp"], default="window")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--label_delay", type=int, default=200)
    parser.add_argument("--max_events", type=int, default=0, help="Limit number of streamed events (0 = no limit)")
    args = parser.parse_args()

    df = load_dataset()
    _, _, _, _, _, _, X_test, y_test = temporal_split(df)
    if args.max_events and args.max_events > 0:
        X_test = X_test.iloc[: args.max_events]
        y_test = y_test.iloc[: args.max_events]
    clf = load("models/model.joblib")
    probs = clf.predict_proba(X_test)[:, 1]

    sim = StreamSimulator(alpha=args.alpha, mode=args.mode, window=args.window, decay=args.decay,
                          label_delay=args.label_delay)
    log = sim.simulate(probs=probs, y_true=y_test.to_numpy())

    out = Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log["idx"], log["coverage"], label="coverage")
    ax.axhline(1.0 - args.alpha, color="red", ls="--", label="target 1-α")
    ax.set_xlabel("events");
    ax.set_ylabel("coverage");
    ax.legend();
    ax.set_title("Streaming coverage")
    fig.tight_layout();
    fig.savefig(out / "stream_coverage.png");
    plt.close(fig)

    # Save CSV + summary
    pd.DataFrame(log).to_csv(out / "streaming_metrics.csv", index=False)
    summary = {
        "alpha": args.alpha,
        "mode": args.mode,
        "window": args.window,
        "decay": args.decay,
        "label_delay": args.label_delay,
        "final_coverage": log["final_coverage"],
        "final_violation_rate": log["final_violation_rate"],
        "artifacts": {
            "stream_coverage": str((out / "stream_coverage.png").as_posix()),
            "streaming_metrics": str((out / "streaming_metrics.csv").as_posix()),
        },
    }
    (out / "stream_summary.json").write_text(json.dumps(summary, indent=2))
    print("Streaming summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
Streaming simulation skeleton with label delays and drift.

Replays events in chronological order, simulates label delays, maintains
sliding-window or exponentially decayed calibration for coverage/violations,
and produces diagnostic plots for ablation studies.
"""

from typing import Any, Dict, Iterator, Tuple


class StreamSimulator:
    """Encapsulates event replay, online calibration, and coverage tracking."""

    def event_stream(self, df: Any, label_delay: float | int, order_by: str) -> Iterator[Dict[str, Any]]:
        """Yield events with features and (delayed) labels according to order."""
        pass

    def update_coverage_window(self, scores: list[float], labels: list[int], alpha: float, window: int) -> Dict[
        str, float]:
        """Update and compute coverage and violation rates within a window."""
        pass

    def inject_drift(self, df: Any, kind: str, severity: float) -> Any:
        """Inject controlled distributional drift for analysis."""
        pass

    def plot_results(self, log: Any, out_dir: str) -> None:
        """Render streaming coverage/violation figures over time."""
        pass
