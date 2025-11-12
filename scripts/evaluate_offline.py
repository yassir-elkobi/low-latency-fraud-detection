from __future__ import annotations

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from pathlib import Path
from typing import Any, Dict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from scripts.common import load_dataset as common_load_dataset, temporal_split as common_temporal_split, \
    build_preprocessor as common_build_preprocessor, compute_metrics as common_compute_metrics
from math import isfinite


class OfflineEvaluator:
    """Generate offline figures (ROC/PR/reliability) and a compact metrics table."""

    # ---------------------------- Data ----------------------------
    # Use shared helpers from scripts/common.py in main()

    # ----------------------- Rebuild base model -----------------------
    def _parse_base_label(self, model_name: str, label: str) -> Dict[str, Any]:
        if model_name == "logreg":
            m = re.search(r"LR\(C=([^\)]+)\)", label)
            if not m:
                raise ValueError(f"Unrecognized LR label: {label}")
            return {"C": float(m.group(1))}
        if model_name == "gbdt":
            m = re.search(r"GBDT\(n=(\d+),lr=([^\)]+)\)", label)
            if not m:
                raise ValueError(f"Unrecognized GBDT label: {label}")
            return {"n_estimators": int(m.group(1)), "learning_rate": float(m.group(2))}
        raise ValueError(f"Unknown model_name: {model_name}")

    def rebuild_base(self, base_meta: Dict[str, Any], pre: ColumnTransformer) -> Pipeline:
        name = base_meta.get("model_name")
        label = base_meta.get("label", "")
        params = self._parse_base_label(name, label)
        if name == "logreg":
            clf = LogisticRegression(
                C=params["C"], penalty="l2", solver="lbfgs", max_iter=2000, class_weight="balanced"
            )
        elif name == "gbdt":
            clf = GradientBoostingClassifier(
                n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], max_depth=3
            )
        else:
            raise ValueError(f"Unsupported base model: {name}")
        return Pipeline([("pre", pre), ("clf", clf)])

    # ---------------------------- Metrics ----------------------------
    def compute_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        return {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "brier": float(brier_score_loss(y_true, y_score)),
            "nll": float(log_loss(y_true, y_score, labels=[0, 1])),
        }

    # ---------------------------- Plots ----------------------------
    def plot_roc_pr(self, y_true: np.ndarray, pre: np.ndarray, post: np.ndarray, out_dir: Path) -> Dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        # ROC
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(y_true, pre, name="pre-proba", ax=ax)
        RocCurveDisplay.from_predictions(y_true, post, name="calibrated-proba", ax=ax)
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        ax.set_title("ROC (probabilities)")
        fig.tight_layout();
        fig.savefig(out_dir / "roc.png");
        plt.close(fig)
        # PR
        fig, ax = plt.subplots(figsize=(5, 5))
        PrecisionRecallDisplay.from_predictions(y_true, pre, name="pre-proba", ax=ax)
        PrecisionRecallDisplay.from_predictions(y_true, post, name="calibrated-proba", ax=ax)
        ax.set_title("PR (probabilities)")
        fig.tight_layout();
        fig.savefig(out_dir / "pr.png");
        plt.close(fig)
        # Histogram
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(pre, bins=30, alpha=0.6, label="pre-proba")
        ax.hist(post, bins=30, alpha=0.6, label="calibrated-proba")
        ax.set_title("Score histogram (probabilities)")
        ax.legend()
        fig.tight_layout();
        fig.savefig(out_dir / "hist_scores.png");
        plt.close(fig)
        return {
            "roc": str((out_dir / "roc.png").as_posix()),
            "pr": str((out_dir / "pr.png").as_posix()),
            "hist": str((out_dir / "hist_scores.png").as_posix()),
        }

    def plot_reliability_pair(self, y_true: np.ndarray, pre: np.ndarray, post: np.ndarray, out_dir: Path) -> Dict[
        str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Pre
        prob_true, prob_pred = calibration_curve(y_true, pre, n_bins=15, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], ls="--", color="gray")
        ax.plot(prob_pred, prob_true, marker="o", label="pre")
        ax.set_title("Reliability (pre)");
        ax.legend(loc="lower right")
        fig.tight_layout();
        fig.savefig(out_dir / "reliability_pre.png");
        plt.close(fig)
        # Post
        prob_true, prob_pred = calibration_curve(y_true, post, n_bins=15, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], ls="--", color="gray")
        ax.plot(prob_pred, prob_true, marker="o", label="post")
        ax.set_title("Reliability (post)");
        ax.legend(loc="lower right")
        fig.tight_layout();
        fig.savefig(out_dir / "reliability_post.png");
        plt.close(fig)
        return {
            "reliability_pre": str((out_dir / "reliability_pre.png").as_posix()),
            "reliability_post": str((out_dir / "reliability_post.png").as_posix()),
        }

    # ------------------------- Metric helpers -------------------------
    def _brier_manual(self, y: np.ndarray, p: np.ndarray) -> float:
        y = y.astype(float)
        p = p.astype(float)
        return float(np.mean((p - y) ** 2))

    def _nll_manual(self, y: np.ndarray, p: np.ndarray) -> float:
        y = y.astype(float)
        p = p.astype(float)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def _compute_metrics_strict(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """Compute ROC/PR and explicitly compute Brier and NLL with sanity checks."""
        # Core metrics
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))
        # Manual
        brier_m = self._brier_manual(y_true, y_score)
        nll_m = self._nll_manual(y_true, y_score)
        # Sklearn refs
        brier_sk = float(brier_score_loss(y_true, y_score))
        nll_sk = float(log_loss(y_true, y_score, labels=[0, 1]))
        # Sanity: they should match closely
        if not (abs(brier_m - brier_sk) < 1e-10 and abs(nll_m - nll_sk) < 1e-8):
            # Prefer manual, but keep awareness
            print(f"[warn] metric mismatch brier(manual={brier_m}, sk={brier_sk}) nll(manual={nll_m}, sk={nll_sk})")
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier": brier_m,
            "nll": nll_m,
        }

    # -------------------------- Orchestrate --------------------------
    def main(self) -> None:
        models_dir = Path("models")
        artifacts_dir = Path("artifacts")
        metrics_path = models_dir / "metrics_offline.json"
        if not metrics_path.exists():
            raise FileNotFoundError("Run training first to create models/metrics_offline.json")
        metrics = json.loads(metrics_path.read_text())

        # Data and split
        df = common_load_dataset()
        X_train, y_train, X_valid, y_valid, X_cal, y_cal, X_test, y_test = common_temporal_split(df)
        numeric_cols = list(X_train.columns)
        pre = common_build_preprocessor(numeric_cols)

        # Rebuild base and fit on train+valid for pre-calibration scores
        base_model = self.rebuild_base(metrics.get("base_model", {}), pre)
        X_trv = pd.concat([X_train, X_valid], axis=0)
        y_trv = pd.concat([y_train, y_valid], axis=0)
        base_model.fit(X_trv, y_trv)
        p_test_pre = base_model.predict_proba(X_test)[:, 1]

        # Load serving calibrated model (may be isotonic or sigmoid) - used for reference only
        calibrated = load(models_dir / "model.joblib")
        _ = calibrated.predict_proba(X_test)[:, 1]  # sanity eval (not used for plots/metrics below)

        # Fit a fresh Platt/sigmoid on the same base (prefit) using calibration split to preserve ranking
        platt = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
        platt.fit(X_cal, y_cal)
        p_test_sigmoid = platt.predict_proba(X_test)[:, 1]

        # Guard: ensure shapes and indices align
        assert p_test_pre.shape == p_test_sigmoid.shape == y_test.shape, "Pre/Post(Test) shapes must align"
        assert np.isfinite(p_test_pre).all() and np.isfinite(p_test_sigmoid).all(), "Probabilities must be finite"
        assert (p_test_pre >= 0).all() and (p_test_pre <= 1).all(), "Pre probabilities out of range"
        assert (p_test_sigmoid >= 0).all() and (p_test_sigmoid <= 1).all(), "Post probabilities out of range"
        # Diagnostics: detect rank-breaking transforms (heavy quantization/rounding)
        n = len(p_test_post)
        uniq_pre = len(np.unique(np.round(p_test_pre.astype(float), 6)))
        uniq_post = len(np.unique(np.round(p_test_post.astype(float), 6)))
        if n > 0:
            ratio_pre = uniq_pre / n
            ratio_post = uniq_post / n
            if ratio_post < 0.02 and ratio_post < ratio_pre / 10:
                print(f"[warn] post probabilities appear highly quantized: unique_ratio_post={ratio_post:.4f} vs pre={ratio_pre:.4f}")
        # Optional: write a small debug CSV for manual inspection
        try:
            dbg = pd.DataFrame({"y": y_test.to_numpy(), "pre": p_test_pre, "post": p_test_post})
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            dbg.sample(min(5000, len(dbg)), random_state=0).to_csv(artifacts_dir / "debug_test_preds_sample.csv", index=False)
        except Exception:
            pass

        # Compute metrics
        y_test_np = y_test.to_numpy()
        computed_pre = self._compute_metrics_strict(y_test_np, p_test_pre)
        computed_post = self._compute_metrics_strict(y_test_np, p_test_sigmoid)

        # Plots
        figs_curves = self.plot_roc_pr(y_test_np, p_test_pre, p_test_sigmoid, artifacts_dir)
        figs_reliab = self.plot_reliability_pair(y_test_np, p_test_pre, p_test_sigmoid, artifacts_dir)

        # Merge and save summary
        merged = {
            **metrics,
            "metrics_pre": computed_pre,
            "metrics_post": computed_post,
            "artifacts": {
                **metrics.get("artifacts", {}),
                **figs_curves,
                **figs_reliab,
            },
        }
        metrics_path.write_text(json.dumps(merged, indent=2))
        print("Offline evaluation artifacts written:", json.dumps(merged.get("artifacts", {}), indent=2))


if __name__ == "__main__":
    OfflineEvaluator().main()
