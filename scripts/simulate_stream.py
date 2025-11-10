from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple
from scripts.common import load_dataset, temporal_split


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

    def effective_n(self) -> float:
        return float(len(self.data))


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

    def effective_n(self) -> float:
        if not self.weights:
            return 0.0
        return float(np.sum(self.weights))


class StreamSimulator:
    """Streaming simulator for online split-conformal with window/decay buffers."""

    def __init__(self, alpha: float = 0.05, mode: str = "window", window: int = 2000, decay: float = 0.01,
                 label_delay: int = 200, warmup: int = 200) -> None:
        self.alpha = alpha
        self.q = 1.0 - alpha
        self.mode = mode
        self.window = window
        self.decay = decay
        self.label_delay = label_delay
        self.warmup = warmup
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
                # Evaluate coverage with the tau that was in effect BEFORE updating with this label
                tau_eval = self.buf.quantile(self.q)
                if np.isnan(tau_eval):
                    in_set = True
                else:
                    in_set = (yj == 1 and pj >= (1.0 - tau_eval)) or (yj == 0 and (1.0 - pj) >= (1.0 - tau_eval))
                # Now update buffer with its nonconformity score
                self.buf.add(s)
                total += 1
                covered += int(in_set)
                violations += int(not in_set)
                if total >= self.warmup:
                    idx_hist.append(total)
                    cov_hist.append(covered / total)
                    viol_hist.append(violations / total)

        # flush
        while pending:
            j, yj = pending.popleft()
            pj = float(probs[j])
            s = pj if yj == 0 else (1.0 - pj)
            tau_eval = self.buf.quantile(self.q)
            if np.isnan(tau_eval):
                in_set = True
            else:
                in_set = (yj == 1 and pj >= (1.0 - tau_eval)) or (yj == 0 and (1.0 - pj) >= (1.0 - tau_eval))
            self.buf.add(s)
            total += 1
            covered += int(in_set)
            violations += int(not in_set)
            if total >= self.warmup:
                idx_hist.append(total)
                cov_hist.append(covered / total)
                viol_hist.append(violations / total)

        return {
            "idx": idx_hist,
            "coverage": cov_hist,
            "violations": viol_hist,
            "final_coverage": cov_hist[-1] if cov_hist else None,
            "final_violation_rate": viol_hist[-1] if viol_hist else None,
            "effective_n": self.buf.effective_n(),
            "warmup": self.warmup,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming Conformal Simulator")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mode", choices=["window", "exp"], default="window")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--label_delay", type=int, default=200)
    parser.add_argument("--max_events", type=int, default=0, help="Limit number of streamed events (0 = no limit)")
    parser.add_argument("--warmup", type=int, default=200, help="Do not report coverage until >= warmup labels")
    # Ablation controls
    parser.add_argument("--ablate", action="store_true", help="Run ablation over window sizes or decays")
    parser.add_argument("--ablate_windows", type=str, default="500,2000,8000", help="Comma-separated window sizes")
    parser.add_argument("--ablate_decays", type=str, default="0.005,0.01,0.02", help="Comma-separated decay values")
    return parser.parse_args()


def load_test_split(max_events: int) -> Tuple[pd.DataFrame, pd.Series]:
    df = load_dataset()
    _, _, _, _, _, _, X_test, y_test = temporal_split(df)
    if max_events and max_events > 0:
        X_test = X_test.iloc[: max_events]
        y_test = y_test.iloc[: max_events]
    return X_test, y_test


def predict_probabilities(X: pd.DataFrame) -> np.ndarray:
    clf = load("models/model.joblib")
    return clf.predict_proba(X)[:, 1]


def plot_coverage(log: Dict[str, Any], alpha: float, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log["idx"], log["coverage"], label="coverage")
    ax.axhline(1.0 - alpha, color="red", ls="--", label="target 1-α")
    ax.set_xlabel("events")
    ax.set_ylabel("coverage")
    ax.legend()
    ax.set_title("Streaming Coverage")
    fig.tight_layout()
    png_path = out_dir / "stream_coverage.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def save_outputs(log: Dict[str, Any], args: argparse.Namespace, out_dir: Path) -> None:
    pd.DataFrame(log).to_csv(out_dir / "streaming_metrics.csv", index=False)
    summary = {
        "alpha": args.alpha,
        "mode": args.mode,
        "window": args.window,
        "decay": args.decay,
        "label_delay": args.label_delay,
        "warmup": args.warmup,
        "final_coverage": log["final_coverage"],
        "final_violation_rate": log["final_violation_rate"],
        "effective_n": log.get("effective_n"),
        "artifacts": {
            "stream_coverage": str((out_dir / "stream_coverage.png").as_posix()),
            "streaming_metrics": str((out_dir / "streaming_metrics.csv").as_posix()),
        },
    }
    (out_dir / "stream_summary.json").write_text(json.dumps(summary, indent=2))
    print("Streaming Summary:", json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    X_test, y_test = load_test_split(args.max_events)
    probs = predict_probabilities(X_test)
    sim = StreamSimulator(alpha=args.alpha, mode=args.mode, window=args.window, decay=args.decay,
                          label_delay=args.label_delay, warmup=args.warmup)
    log = sim.simulate(probs=probs, y_true=y_test.to_numpy())
    out = Path("artifacts")
    plot_coverage(log, args.alpha, out)
    save_outputs(log, args, out)
    # Optional ablation
    if args.ablate:
        records: List[Dict[str, Any]] = []
        if args.mode == "window":
            vals = [int(v) for v in str(args.ablate_windows).split(",") if v.strip()]
            for w in vals:
                sim_w = StreamSimulator(alpha=args.alpha, mode="window", window=w, decay=args.decay,
                                        label_delay=args.label_delay, warmup=args.warmup)
                log_w = sim_w.simulate(probs=probs, y_true=y_test.to_numpy())
                records.append({
                    "mode": "window",
                    "param": "W",
                    "value": w,
                    "final_coverage": log_w.get("final_coverage"),
                    "final_violation_rate": log_w.get("final_violation_rate"),
                    "effective_n": log_w.get("effective_n"),
                    "warmup": args.warmup,
                    "label_delay": args.label_delay,
                    "alpha": args.alpha,
                })
        else:
            vals = [float(v) for v in str(args.ablate_decays).split(",") if v.strip()]
            for d in vals:
                sim_d = StreamSimulator(alpha=args.alpha, mode="exp", window=args.window, decay=d,
                                        label_delay=args.label_delay, warmup=args.warmup)
                log_d = sim_d.simulate(probs=probs, y_true=y_test.to_numpy())
                records.append({
                    "mode": "exp",
                    "param": "lambda",
                    "value": d,
                    "final_coverage": log_d.get("final_coverage"),
                    "final_violation_rate": log_d.get("final_violation_rate"),
                    "effective_n": log_d.get("effective_n"),
                    "warmup": args.warmup,
                    "label_delay": args.label_delay,
                    "alpha": args.alpha,
                })
        if records:
            df = pd.DataFrame(records)
            df.to_csv(out / "stream_ablation.csv", index=False)
            (out / "stream_ablation.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
