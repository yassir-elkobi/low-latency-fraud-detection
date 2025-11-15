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

"""
Streaming conformal prediction simulator for the /predict microservice.

Provides:
- Online split-conformal simulation with label-conditional (Mondrian) thresholds
- Windowed or exponential-decay buffers
- Label delay and warmup handling
- Optional ablation sweeps over window/decay and alpha, plus simple segmentations

Outputs:
- artifacts/stream_coverage.png, artifacts/streaming_metrics.csv, artifacts/stream_summary.json
- Optional: artifacts/stream_ablation.csv and stream_ablation.json
"""


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

    def conformal_tau(self, q: float) -> float:
        """Finite-sample conservative quantile using order statistic ceil((n+1)q)."""
        if not self.data:
            return float("nan")
        arr = np.sort(np.asarray(self.data, dtype=float))
        n = len(arr)
        # Index per split-conformal finite-sample guarantee
        k = int(np.ceil((n + 1) * q)) - 1
        k = min(max(k, 0), n - 1)
        return float(arr[k])

    def conformal_tau_with_index(self, q: float) -> Tuple[int, float]:
        """Return (index, tau) for the conservative quantile ceil((n+1)q)."""
        if not self.data:
            return (-1, float("nan"))
        arr = np.sort(np.asarray(self.data, dtype=float))
        n = len(arr)
        k = int(np.ceil((n + 1) * q)) - 1
        k = min(max(k, 0), n - 1)
        return (int(k), float(arr[k]))

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

    def conformal_tau(self, q: float) -> float:
        """Conservative weighted quantile using cutoff ceil((W+1)q)."""
        if not self.values:
            return float("nan")
        v = np.asarray(self.values, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        order = np.argsort(v)
        v = v[order]
        w = w[order]
        cw = np.cumsum(w)
        W = cw[-1]
        cutoff = np.ceil((W + 1.0) * q)
        idx = int(np.searchsorted(cw, cutoff, side="left"))
        idx = min(idx, len(v) - 1)
        return float(v[idx])

    def conformal_tau_with_index(self, q: float) -> Tuple[int, float]:
        """Return (index, tau) using weighted cutoff ceil((W+1)q)."""
        if not self.values:
            return (-1, float("nan"))
        v = np.asarray(self.values, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        order = np.argsort(v)
        v = v[order]
        w = w[order]
        cw = np.cumsum(w)
        W = cw[-1]
        cutoff = np.ceil((W + 1.0) * q)
        idx = int(np.searchsorted(cw, cutoff, side="left"))
        idx = min(idx, len(v) - 1)
        return (int(idx), float(v[idx]))


class StreamSimulator:
    """Streaming simulator for online split-conformal with window/decay buffers."""

    def __init__(self, alpha: float = 0.05, mode: str = "window", window: int = 2000, decay: float = 0.01,
                 label_delay: int = 200, warmup: int = 200, alpha_pos: float | None = None,
                 mondrian_seg: str = "label",
                 alpha_neg: float | None = None,
                 adapt_gain: float = 0.4,
                 adapt_gain_pos: float = 0.9,
                 adapt_start_pos_n: int = 100,
                 pos_slack: float = 0.0) -> None:
        self.alpha = alpha
        self.q = 1.0 - alpha
        self.alpha_pos_override = alpha_pos
        self.alpha_neg_override = alpha_neg
        self.mode = mode
        self.window = window
        self.decay = decay
        self.label_delay = label_delay
        self.warmup = warmup
        self.mondrian_seg = mondrian_seg  # "label", "label_amount", "label_hour"
        self.adapt_gain = float(adapt_gain)
        self.adapt_gain_pos = float(adapt_gain_pos)
        self.adapt_start_pos_n = int(adapt_start_pos_n)
        self.pos_slack = max(0.0, float(pos_slack))
        # Buffers: either single per label, or per (label, segment)
        self.buf_pos = WindowBuf(window) if mode == "window" else ExpBuf(decay)
        self.buf_neg = WindowBuf(window) if mode == "window" else ExpBuf(decay)
        self.buf_pos_map: Dict[int, WindowBuf | ExpBuf] = {}
        self.buf_neg_map: Dict[int, WindowBuf | ExpBuf] = {}

    def _get_pos_buf(self, seg_id: int) -> WindowBuf | ExpBuf:
        if self.mondrian_seg == "label":
            return self.buf_pos
        if seg_id not in self.buf_pos_map:
            self.buf_pos_map[seg_id] = (WindowBuf(self.window) if self.mode == "window" else ExpBuf(self.decay))
        return self.buf_pos_map[seg_id]

    def _get_neg_buf(self, seg_id: int) -> WindowBuf | ExpBuf:
        if self.mondrian_seg == "label":
            return self.buf_neg
        if seg_id not in self.buf_neg_map:
            self.buf_neg_map[seg_id] = (WindowBuf(self.window) if self.mode == "window" else ExpBuf(self.decay))
        return self.buf_neg_map[seg_id]

    def _q_pos(self) -> float:
        """Slight positive-class slack to raise coverage under class scarcity."""
        # Use segment-local buffer if segmented; default to global pos buffer
        if self.mondrian_seg == "label":
            npos_eff = self.buf_pos.effective_n()
        else:
            # Conservative: use minimum effective_n across segments to avoid overconfidence
            vals = [b.effective_n() for b in self.buf_pos_map.values()] or [0.0]
            npos_eff = float(min(vals))
        if self.pos_slack <= 0.0:
            return min(0.999999, self.q)
        slack = min(self.pos_slack, 1.0 / (max(1.0, npos_eff) + 1.0))
        return min(0.999999, self.q + slack)

    def simulate(self, probs: np.ndarray, y_true: np.ndarray, segments: np.ndarray | None = None) -> Dict[str, Any]:
        """Run the streaming loop and return logs and final summary metrics."""
        state = self._init_state()
        if segments is None:
            segments = np.zeros_like(probs, dtype=int)
        for i, _ in enumerate(probs):
            state["pending"].append((i, int(y_true[i])))
            if len(state["pending"]) > self.label_delay:
                j, yj = state["pending"].popleft()
                pj = float(probs[j])
                seg_j = int(segments[j]) if segments is not None else 0
                state["alpha_pos_eff"] = self._maybe_update_alpha_pos(state)
                in_pos, in_neg, set_size = self._decide_set_membership(pj, seg_j)
                in_set = (yj == 1 and in_pos) or (yj == 0 and in_neg)
                self._update_after_label(state, yj, pj, seg_j, in_set, set_size)
                self._instrument_tick(state)
                self._maybe_append_histories(state)
        self._flush_pending(state, probs, segments)
        return self._finalize(state)

    # ------------- helpers: simulation internals -------------
    def _init_state(self) -> Dict[str, Any]:
        return {
            "pending": deque(),  # type: Deque[Tuple[int, int]]
            "total": 0,
            "covered": 0,
            "total_pos": 0,
            "covered_pos": 0,
            "total_neg": 0,
            "covered_neg": 0,
            "sum_set_size": 0.0,
            "ambiguous": 0,
            "alpha_pos_eff": (self.alpha_pos_override if self.alpha_pos_override is not None else self.alpha),
            "pos_adapt_started_total": None,  # type: int | None
            "idx_hist": [],
            "cov_hist": [],
            "cov_pos_hist": [],
            "cov_neg_hist": [],
            "viol_hist": [],
        }

    def _maybe_update_alpha_pos(self, state: Dict[str, Any]) -> float:
        if state["total_pos"] >= self.adapt_start_pos_n and self.alpha_pos_override is None:
            cov_pos_so_far = (state["covered_pos"] / state["total_pos"]) if state["total_pos"] > 0 else 0.0
            gap = (1.0 - self.alpha) - cov_pos_so_far
            new_alpha = float(min(0.5, max(1e-6, self.alpha - self.adapt_gain_pos * gap)))
            if state["pos_adapt_started_total"] is None:
                state["pos_adapt_started_total"] = state["total"]
            return new_alpha
        return state["alpha_pos_eff"]

    def _thresholds(self, seg_j: int) -> Tuple[float, float]:
        pos_buf = self._get_pos_buf(seg_j)
        neg_buf = self._get_neg_buf(seg_j)
        tau_pos = pos_buf.conformal_tau(self._q_pos())
        alpha_neg_eff = self.alpha_neg_override if self.alpha_neg_override is not None else self.alpha
        q_neg_eff = min(0.999999, 1.0 - alpha_neg_eff)
        tau_neg = neg_buf.conformal_tau(q_neg_eff)
        return tau_pos, tau_neg

    def _decide_set_membership(self, pj: float, seg_j: int) -> Tuple[bool, bool, int]:
        tau_pos, tau_neg = self._thresholds(seg_j)
        if np.isnan(tau_pos) or np.isnan(tau_neg):
            in_pos = True
            in_neg = True
        else:
            in_pos = pj >= (1.0 - tau_pos)
            in_neg = (1.0 - pj) >= (1.0 - tau_neg)
        set_size = int(in_pos) + int(in_neg)
        if set_size == 0:
            in_pos = (1.0 - pj) <= pj
            in_neg = not in_pos
            set_size = 1
        return in_pos, in_neg, set_size

    def _update_after_label(self, state: Dict[str, Any], yj: int, pj: float, seg_j: int, in_set: bool,
                            set_size: int) -> None:
        if set_size > 1:
            state["ambiguous"] += 1
        s = (1.0 - pj) if yj == 1 else pj
        if yj == 1:
            self._get_pos_buf(seg_j).add(s)
        else:
            self._get_neg_buf(seg_j).add(s)
        state["total"] += 1
        state["covered"] += int(in_set)
        state["sum_set_size"] += float(set_size)
        if yj == 1:
            state["total_pos"] += 1
            state["covered_pos"] += int(in_set)
        else:
            state["total_neg"] += 1
            state["covered_neg"] += int(in_set)

    def _instrument_tick(self, state: Dict[str, Any]) -> None:
        total = state["total"]
        if (total > 0) and (total % 1000 == 0):
            if self.mondrian_seg == "label":
                n_eff_pos = self.buf_pos.effective_n()
                n_eff_neg = self.buf_neg.effective_n()
                tau_pos_now = self.buf_pos.conformal_tau(self._q_pos())
                alpha_neg_eff = self.alpha_neg_override if self.alpha_neg_override is not None else self.alpha
                q_neg_eff_now = min(0.999999, 1.0 - alpha_neg_eff)
                tau_neg_now = self.buf_neg.conformal_tau(q_neg_eff_now)
            else:
                n_eff_pos = sum(b.effective_n() for b in self.buf_pos_map.values())
                n_eff_neg = sum(b.effective_n() for b in self.buf_neg_map.values())
                tau_pos_now = float("nan")
                tau_neg_now = float("nan")
            print(json.dumps({
                "tick": total,
                "n_pos_labeled": state["total_pos"],
                "n_eff_pos": n_eff_pos,
                "n_eff_neg": n_eff_neg,
                "tau_pos": tau_pos_now,
                "tau_neg": tau_neg_now,
                "alpha_pos_eff": state["alpha_pos_eff"],
            }))

    def _maybe_append_histories(self, state: Dict[str, Any]) -> None:
        total = state["total"]
        if total >= self.wwarmup if hasattr(self, "wwarmup") else self.warmup:
            # Use self.warmup; the hasattr guard is a no-op placeholder to avoid logic change
            total = state["total"]
            covered = state["covered"]
            total_pos = state["total_pos"]
            total_neg = state["total_neg"]
            state["idx_hist"].append(total)
            state["cov_hist"].append(covered / total)
            state["viol_hist"].append(1.0 - (covered / total))
            state["cov_pos_hist"].append((state["covered_pos"] / total_pos) if total_pos > 0 else float("nan"))
            state["cov_neg_hist"].append((state["covered_neg"] / total_neg) if total_neg > 0 else float("nan"))

    def _flush_pending(self, state: Dict[str, Any], probs: np.ndarray, segments: np.ndarray | None) -> None:
        while state["pending"]:
            j, yj = state["pending"].popleft()
            pj = float(probs[j])
            seg_j = int(segments[j]) if segments is not None else 0
            in_pos, in_neg, set_size = self._decide_set_membership(pj, seg_j)
            in_set = (yj == 1 and in_pos) or (yj == 0 and in_neg)
            self._update_after_label(state, yj, pj, seg_j, in_set, set_size)
            # history update (no instrumentation during flush)
            if state["total"] >= self.warmup:
                total = state["total"]
                covered = state["covered"]
                total_pos = state["total_pos"]
                total_neg = state["total_neg"]
                state["idx_hist"].append(total)
                state["cov_hist"].append(covered / total)
                state["viol_hist"].append(1.0 - (covered / total))
                state["cov_pos_hist"].append((state["covered_pos"] / total_pos) if total_pos > 0 else float("nan"))
                state["cov_neg_hist"].append((state["covered_neg"] / total_neg) if total_neg > 0 else float("nan"))

    def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        cov_hist = state["cov_hist"]
        cov_pos_hist = state["cov_pos_hist"]
        cov_neg_hist = state["cov_neg_hist"]
        final_cov = cov_hist[-1] if cov_hist else None
        final_cov_pos = cov_pos_hist[-1] if cov_pos_hist else None
        final_cov_neg = cov_neg_hist[-1] if cov_neg_hist else None
        avg_set_size = (state["sum_set_size"] / state["total"]) if state["total"] > 0 else None
        ambiguous_share = (state["ambiguous"] / state["total"]) if state["total"] > 0 else None
        if self.mondrian_seg == "label":
            eff_n = (self.buf_pos.effective_n() + self.buf_neg.effective_n())
        else:
            eff_n = 0.0
            for b in self.buf_pos_map.values():
                eff_n += b.effective_n()
            for b in self.buf_neg_map.values():
                eff_n += b.effective_n()
        final_violation_gap = None
        if final_cov is not None:
            final_violation_gap = max(0.0, (1.0 - self.alpha) - float(final_cov))
        return {
            "idx": state["idx_hist"],
            "coverage": cov_hist,
            "coverage_pos": cov_pos_hist,
            "coverage_neg": cov_neg_hist,
            "violations": state["viol_hist"],
            "final_coverage": final_cov,
            "final_coverage_pos": final_cov_pos,
            "final_coverage_neg": final_cov_neg,
            "final_violation_rate": final_violation_gap,
            "avg_set_size": avg_set_size,
            "effective_n": eff_n,
            "ambiguous_share": ambiguous_share,
            "warmup": self.warmup,
            "alpha_pos_eff": state["alpha_pos_eff"],
            "pos_adapt_started_total": state["pos_adapt_started_total"],
        }


def parse_args() -> argparse.Namespace:
    """CLI flags for the simulator and ablations."""
    parser = argparse.ArgumentParser(description="Streaming Conformal Simulator")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha_pos", type=float, default=None, help="Optional stricter alpha for fraud class")
    parser.add_argument("--alpha_neg", type=float, default=None, help="Optional alpha for non-fraud class")
    parser.add_argument("--mode", choices=["window", "exp"], default="window")
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--label_delay", type=int, default=200)
    parser.add_argument("--max_events", type=int, default=0, help="Limit number of streamed events (0 = no limit)")
    parser.add_argument("--warmup", type=int, default=200, help="Do not report coverage until >= warmup labels")
    parser.add_argument("--adapt_gain", type=float, default=0.4, help="Adaptation gain for alpha_pos (0-1)")
    parser.add_argument("--adapt_gain_pos", type=float, default=0.9, help="Positive-class adaptation gain (0-1)")
    parser.add_argument("--adapt_start_pos_n", type=int, default=100, help="Start adapting after N positive labels")
    parser.add_argument("--pos_slack", type=float, default=0.0, help="Max extra slack added to q_pos (0 to disable)")
    # Ablation controls
    parser.add_argument("--ablate", action="store_true", help="Run ablation over window sizes or decays")
    parser.add_argument("--ablate_windows", type=str, default="500,2000,8000", help="Comma-separated window sizes")
    parser.add_argument("--ablate_decays", type=str, default="0.005,0.01,0.02", help="Comma-separated decay values")
    parser.add_argument("--ablate_alphas", type=str, default="0.10,0.05,0.01",
                        help="Comma-separated alpha (miscoverage) values")
    parser.add_argument("--mondrian_seg", choices=["label", "label_amount", "label_hour"], default="label",
                        help="Mondrian segmentation: label-only, or label+amount tertiles, or label+time-of-day")
    return parser.parse_args()


def load_test_split(max_events: int) -> Tuple[pd.DataFrame, pd.Series]:
    """Load held-out test split and optionally cap the number of events."""
    df = load_dataset()
    _, _, _, _, _, _, X_test, y_test = temporal_split(df)
    if max_events and max_events > 0:
        X_test = X_test.iloc[: max_events]
        y_test = y_test.iloc[: max_events]
    return X_test, y_test


def predict_probabilities(X: pd.DataFrame) -> np.ndarray:
    """Load serving model and produce probabilities on X."""
    clf = load("models/model.joblib")
    return clf.predict_proba(X)[:, 1]


def plot_coverage(log: Dict[str, Any], alpha: float, out_dir: Path) -> Path:
    """Save coverage vs events plot; return the PNG path."""
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


def save_outputs(log: Dict[str, Any], args: argparse.Namespace, out_dir: Path,
                 seg_desc: Dict[str, Any] | None = None) -> None:
    """Persist streaming log CSV and a compact JSON summary with artifact paths."""
    pd.DataFrame(log).to_csv(out_dir / "streaming_metrics.csv", index=False)
    target = 1.0 - args.alpha
    summary = {
        "alpha": args.alpha,
        "alpha_pos": args.alpha_pos,
        "mode": args.mode,
        "window": args.window,
        "decay": args.decay,
        "label_delay": args.label_delay,
        "warmup": args.warmup,
        "mondrian_seg": args.mondrian_seg,
        "seg_desc": seg_desc or {},
        "final_coverage": log["final_coverage"],
        "final_coverage_pos": log.get("final_coverage_pos"),
        "final_coverage_neg": log.get("final_coverage_neg"),
        "final_violation_rate": log["final_violation_rate"],
        "avg_set_size": log.get("avg_set_size"),
        "effective_n": log.get("effective_n"),
        "ambiguous_share": log.get("ambiguous_share"),
        "delta_coverage": (None if log["final_coverage"] is None else float(log["final_coverage"] - target)),
        "delta_coverage_pos": (
            None if log.get("final_coverage_pos") is None else float(log["final_coverage_pos"] - target)),
        "delta_coverage_neg": (
            None if log.get("final_coverage_neg") is None else float(log["final_coverage_neg"] - target)),
        "pos_adapt_started_total": log.get("pos_adapt_started_total"),
        "artifacts": {
            "stream_coverage": str((out_dir / "stream_coverage.png").as_posix()),
            "streaming_metrics": str((out_dir / "streaming_metrics.csv").as_posix()),
        },
    }
    (out_dir / "stream_summary.json").write_text(json.dumps(summary, indent=2))
    print("Streaming Summary:", json.dumps(summary, indent=2))


def compute_segments(seg_mode: str, X: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute optional segmentation arrays for Mondrian variants and a short descriptor."""
    n = len(X)
    desc: Dict[str, Any] = {}
    if seg_mode == "label":
        return np.zeros(n, dtype=int), {"type": "none"}
    if seg_mode == "label_amount" and "Amount" in X.columns:
        a = X["Amount"].to_numpy(dtype=float)
        q33 = float(np.quantile(a, 0.33))
        q66 = float(np.quantile(a, 0.66))
        bins = np.array([q33, q66], dtype=float)
        seg = np.digitize(a, bins=bins, right=False)  # 0: low, 1: med, 2: high
        return seg.astype(int), {"type": "amount_tertiles", "q33": q33, "q66": q66}
    if seg_mode == "label_hour" and "Time" in X.columns:
        t = X["Time"].to_numpy(dtype=float)
        hours = ((t // 3600) % 24).astype(int)
        # 0: night [0-7], 1: day [8-15], 2: eve [16-23]
        seg = np.where(hours < 8, 0, np.where(hours < 16, 1, 2))
        return seg.astype(int), {"type": "hour_bins", "bins": "[0-7],[8-15],[16-23]"}
    # Fallback: single segment
    return np.zeros(n, dtype=int), {"type": "none"}


def build_simulator_from_args(args: argparse.Namespace, **overrides: Any) -> StreamSimulator:
    """Construct a StreamSimulator instance from CLI args with optional field overrides."""
    params = dict(
        alpha=args.alpha,
        mode=args.mode,
        window=args.window,
        decay=args.decay,
        label_delay=args.label_delay,
        warmup=args.warmup,
        alpha_pos=args.alpha_pos,
        alpha_neg=args.alpha_neg,
        mondrian_seg=args.mondrian_seg,
        adapt_gain=args.adapt_gain,
        adapt_gain_pos=args.adapt_gain_pos,
        adapt_start_pos_n=args.adapt_start_pos_n,
        pos_slack=args.pos_slack,
    )
    params.update(overrides)
    return StreamSimulator(**params)  # type: ignore[arg-type]


def run_single_simulation(args: argparse.Namespace, X_test: pd.DataFrame, y_test: pd.Series, probs: np.ndarray) -> Dict[
    str, Any]:
    """Run the primary simulation, plot, and save outputs. Returns the log dict."""
    seg_main, seg_desc = compute_segments(args.mondrian_seg, X_test)
    sim = build_simulator_from_args(args)
    log = sim.simulate(probs=probs, y_true=y_test.to_numpy(), segments=seg_main)
    out = Path("artifacts")
    plot_coverage(log, args.alpha, out)
    save_outputs(log, args, out, seg_desc=seg_desc)
    return log


def run_ablation(args: argparse.Namespace, X_test: pd.DataFrame, y_test: pd.Series, probs: np.ndarray) -> None:
    """Execute ablation sweeps (window/decay/alpha and simple segmentation), saving CSV/JSON."""
    out = Path("artifacts")
    records: List[Dict[str, Any]] = []
    # Sweep windows / decays
    if args.mode == "window":
        vals = [int(v) for v in str(args.ablate_windows).split(",") if v.strip()]
        for w in vals:
            sim_w = build_simulator_from_args(args, mode="window", window=w, alpha_pos=args.alpha, alpha_neg=args.alpha,
                                              mondrian_seg="label")
            log_w = sim_w.simulate(probs=probs, y_true=y_test.to_numpy())
            records.append({
                "mode": "window",
                "param": "W",
                "value": w,
                "final_coverage": log_w.get("final_coverage"),
                "final_violation_rate": log_w.get("final_violation_rate"),
                "final_coverage_pos": log_w.get("final_coverage_pos"),
                "avg_set_size": log_w.get("avg_set_size"),
                "effective_n": log_w.get("effective_n"),
                "warmup": args.warmup,
                "label_delay": args.label_delay,
                "alpha": args.alpha,
            })
        # Additionally, include a couple of exp-decay rows to check robustness
        decays_extra = [float(v) for v in str(args.ablate_decays).split(",") if v.strip()]
        for d in decays_extra:
            sim_d = build_simulator_from_args(args, mode="exp", decay=d, alpha_pos=args.alpha, alpha_neg=args.alpha,
                                              mondrian_seg="label")
            log_d = sim_d.simulate(probs=probs, y_true=y_test.to_numpy())
            records.append({
                "mode": "exp",
                "param": "lambda",
                "value": d,
                "final_coverage": log_d.get("final_coverage"),
                "final_violation_rate": log_d.get("final_violation_rate"),
                "final_coverage_pos": log_d.get("final_coverage_pos"),
                "avg_set_size": log_d.get("avg_set_size"),
                "effective_n": log_d.get("effective_n"),
                "warmup": args.warmup,
                "label_delay": args.label_delay,
                "alpha": args.alpha,
            })
    else:
        vals = [float(v) for v in str(args.ablate_decays).split(",") if v.strip()]
        for d in vals:
            sim_d = build_simulator_from_args(args, mode="exp", decay=d, alpha_pos=args.alpha, alpha_neg=args.alpha,
                                              mondrian_seg="label")
            log_d = sim_d.simulate(probs=probs, y_true=y_test.to_numpy())
            records.append({
                "mode": "exp",
                "param": "lambda",
                "value": d,
                "final_coverage": log_d.get("final_coverage"),
                "final_violation_rate": log_d.get("final_violation_rate"),
                "final_coverage_pos": log_d.get("final_coverage_pos"),
                "avg_set_size": log_d.get("avg_set_size"),
                "effective_n": log_d.get("effective_n"),
                "warmup": args.warmup,
                "label_delay": args.label_delay,
                "alpha": args.alpha,
            })
    # Alpha sweep (disable adaptation by setting alpha_pos/neg equal)
    alpha_vals_raw = [float(v) for v in str(args.ablate_alphas).split(",") if v.strip()]
    for a_input in alpha_vals_raw:
        # Interpret values > 0.5 as target coverage; convert to miscoverage
        alpha_mis = a_input if a_input <= 0.5 else (1.0 - a_input)
        target_cov = 1.0 - alpha_mis
        sim_a = build_simulator_from_args(args, alpha=alpha_mis, alpha_pos=alpha_mis, alpha_neg=alpha_mis,
                                          mondrian_seg="label")
        log_a = sim_a.simulate(probs=probs, y_true=y_test.to_numpy(), segments=np.zeros_like(probs, dtype=int))
        # Compute label-conditional positive-class conservative quantile and index at the end
        q_pos = min(0.999999, 1.0 - alpha_mis)
        k_pos, tau_pos = sim_a.buf_pos.conformal_tau_with_index(q_pos)
        n_eff_pos = sim_a.buf_pos.effective_n()
        records.append({
            "mode": args.mode,
            "param": "alpha",
            "value": a_input,
            "alpha_used": alpha_mis,
            "target": target_cov,
            "final_coverage": log_a.get("final_coverage"),
            "final_violation_rate": log_a.get("final_violation_rate"),
            "final_coverage_pos": log_a.get("final_coverage_pos"),
            "avg_set_size": log_a.get("avg_set_size"),
            "effective_n": log_a.get("effective_n"),
            "ambiguous_share": log_a.get("ambiguous_share"),
            "alpha_pos_eff": log_a.get("alpha_pos_eff"),
            "q_index_pos": int(k_pos),
            "q_value_pos": float(tau_pos),
            "n_eff_pos": float(n_eff_pos),
            "warmup": args.warmup,
            "label_delay": args.label_delay,
        })
    # Additional segmentation line
    seg_rows: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
    if "Amount" in X_test.columns:
        seg_arr, seg_info = compute_segments("label_amount", X_test)
        seg_rows.append(("label_amount", seg_arr, seg_info))
    elif "Time" in X_test.columns:
        seg_arr, seg_info = compute_segments("label_hour", X_test)
        seg_rows.append(("label_hour", seg_arr, seg_info))
    for seg_name, seg_arr, seg_info in seg_rows:
        sim_seg = build_simulator_from_args(args, mondrian_seg=seg_name, alpha_neg=args.alpha)
        log_seg = sim_seg.simulate(probs=probs, y_true=y_test.to_numpy(), segments=seg_arr)
        records.append({
            "mode": args.mode,
            "param": "seg",
            "value": seg_name,
            "final_coverage": log_seg.get("final_coverage"),
            "final_violation_rate": log_seg.get("final_violation_rate"),
            "final_coverage_pos": log_seg.get("final_coverage_pos"),
            "avg_set_size": log_seg.get("avg_set_size"),
            "effective_n": log_seg.get("effective_n"),
            "ambiguous_share": log_seg.get("ambiguous_share"),
            "warmup": args.warmup,
            "label_delay": args.label_delay,
            "alpha": args.alpha,
            "seg_desc": seg_info,
        })
    if records:
        df = pd.DataFrame(records)
        df.to_csv(out / "stream_ablation.csv", index=False)
        (out / "stream_ablation.json").write_text(json.dumps(records, indent=2))


def main() -> None:
    """CLI entrypoint: run a single simulation, then optional ablation sweeps."""
    args = parse_args()
    X_test, y_test = load_test_split(args.max_events)
    probs = predict_probabilities(X_test)
    _ = run_single_simulation(args, X_test, y_test, probs)
    if args.ablate:
        run_ablation(args, X_test, y_test, probs)


if __name__ == "__main__":
    main()
