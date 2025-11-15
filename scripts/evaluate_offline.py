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
    build_preprocessor as common_build_preprocessor

"""
Offline evaluation pipeline for the /predict microservice.

Generates held-out test-set metrics and plots for both pre-calibration and
post-calibration probabilities:
- ROC and PR curves (labels: "pre-proba" vs "calibrated-proba")
- Reliability diagrams (pre vs post)
- Score histogram

Calibration approach:
- Fit a fresh Platt/sigmoid calibrator (cv="prefit") on the calibration split
  using the same base model to preserve ranking and avoid leakage.
- Evaluate all final metrics on the test split only.

Outputs:
- Artifacts saved in artifacts/: roc.png, pr.png, reliability_pre.png, reliability_post.png, hist_scores.png, and debug_test_preds_sample.csv
- Updates models/metrics_offline.json with metrics_pre/metrics_post and artifact paths for the dashboard/README.
"""


class OfflineEvaluator:

    def _parse_base_label(self, model_name: str, label: str) -> Dict[str, Any]:
        """Parse a compact model label back into hyperparameters for reproducibility."""
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
        """Rebuild the base sklearn Pipeline (preprocessor + classifier) from metadata."""
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

    def compute_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """Compute ROC-AUC, PR-AUC, Brier, and NLL using sklearn primitives."""
        return {
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "brier": float(brier_score_loss(y_true, y_score)),
            "nll": float(log_loss(y_true, y_score, labels=[0, 1])),
        }

    def plot_roc_pr(self, y_true: np.ndarray, pre: np.ndarray, post: np.ndarray, out_dir: Path) -> Dict[str, str]:
        """Create ROC and PR curves and a score histogram; return saved paths."""
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
        """Create reliability diagrams for pre and post probabilities; return saved paths."""
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

    def _brier_manual(self, y: np.ndarray, p: np.ndarray) -> float:
        """Manual Brier score: mean squared error between probabilities and labels."""
        y = y.astype(float)
        p = p.astype(float)
        return float(np.mean((p - y) ** 2))

    def _nll_manual(self, y: np.ndarray, p: np.ndarray) -> float:
        """Manual negative log-likelihood (log loss) with numeric clipping."""
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

    def main(self) -> None:
        """End-to-end evaluation: load data/models, compute metrics, emit artifacts and JSON."""
        models_dir, artifacts_dir, metrics_path = self._paths()
        metrics = self._load_metrics(metrics_path)
        X_train, y_train, X_valid, y_valid, X_cal, y_cal, X_test, y_test, pre = self._prepare_data_and_preprocessor()
        base_model, p_test_pre = self._fit_base_and_predict_pre(metrics, pre, X_train, y_train, X_valid, y_valid,
                                                                X_test)
        self._load_serving_calibrated_reference(models_dir, X_test)
        p_test_sigmoid = self._fit_platt_and_predict_post(base_model, X_cal, y_cal, X_test)
        self._validate_probabilities(p_test_pre, p_test_sigmoid, y_test)
        self._quantization_diagnostics(p_test_pre, p_test_sigmoid)
        self._write_debug_sample(artifacts_dir, y_test, p_test_pre, p_test_sigmoid)
        computed_pre, computed_post = self._compute_all_metrics(y_test, p_test_pre, p_test_sigmoid)
        figs_curves, figs_reliab = self._generate_all_plots(y_test, p_test_pre, p_test_sigmoid, artifacts_dir)
        self._merge_and_save_summary(metrics, computed_pre, computed_post, figs_curves, figs_reliab, metrics_path)
        print("Offline evaluation artifacts written:", json.dumps({**figs_curves, **figs_reliab}, indent=2))

    # ------------- helpers: orchestration -------------
    def _paths(self):
        models_dir = Path("models")
        artifacts_dir = Path("artifacts")
        metrics_path = models_dir / "metrics_offline.json"
        return models_dir, artifacts_dir, metrics_path

    def _load_metrics(self, metrics_path: Path) -> Dict[str, Any]:
        if not metrics_path.exists():
            raise FileNotFoundError("Run training first to create models/metrics_offline.json")
        return json.loads(metrics_path.read_text())

    def _prepare_data_and_preprocessor(self):
        df = common_load_dataset()
        X_train, y_train, X_valid, y_valid, X_cal, y_cal, X_test, y_test = common_temporal_split(df)
        numeric_cols = list(X_train.columns)
        pre = common_build_preprocessor(numeric_cols)
        return X_train, y_train, X_valid, y_valid, X_cal, y_cal, X_test, y_test, pre

    def _fit_base_and_predict_pre(
            self,
            metrics: Dict[str, Any],
            pre: ColumnTransformer,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_valid: pd.DataFrame,
            y_valid: pd.Series,
            X_test: pd.DataFrame,
    ):
        base_model = self.rebuild_base(metrics.get("base_model", {}), pre)
        X_trv = pd.concat([X_train, X_valid], axis=0)
        y_trv = pd.concat([y_train, y_valid], axis=0)
        base_model.fit(X_trv, y_trv)
        p_test_pre = base_model.predict_proba(X_test)[:, 1]
        return base_model, p_test_pre

    def _load_serving_calibrated_reference(self, models_dir: Path, X_test: pd.DataFrame) -> None:
        calibrated = load(models_dir / "model.joblib")
        _ = calibrated.predict_proba(X_test)[:, 1]  # sanity eval (not used for plots/metrics below)

    def _fit_platt_and_predict_post(
            self,
            base_model: Pipeline,
            X_cal: pd.DataFrame,
            y_cal: pd.Series,
            X_test: pd.DataFrame,
    ) -> np.ndarray:
        platt = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
        platt.fit(X_cal, y_cal)
        return platt.predict_proba(X_test)[:, 1]

    def _validate_probabilities(self, p_test_pre: np.ndarray, p_test_sigmoid: np.ndarray, y_test: pd.Series) -> None:
        assert p_test_pre.shape == p_test_sigmoid.shape == y_test.shape, "Pre/Post(Test) shapes must align"
        assert np.isfinite(p_test_pre).all() and np.isfinite(p_test_sigmoid).all(), "Probabilities must be finite"
        assert (p_test_pre >= 0).all() and (p_test_pre <= 1).all(), "Pre probabilities out of range"
        assert (p_test_sigmoid >= 0).all() and (p_test_sigmoid <= 1).all(), "Post probabilities out of range"

    def _quantization_diagnostics(self, p_test_pre: np.ndarray, p_test_sigmoid: np.ndarray) -> None:
        n = len(p_test_sigmoid)
        uniq_pre = len(np.unique(np.round(p_test_pre.astype(float), 6)))
        uniq_post = len(np.unique(np.round(p_test_sigmoid.astype(float), 6)))
        if n > 0:
            ratio_pre = uniq_pre / n
            ratio_post = uniq_post / n
            if ratio_post < 0.02 and ratio_post < ratio_pre / 10:
                print(
                    f"[warn] post probabilities appear highly quantized: unique_ratio_post={ratio_post:.4f} vs pre={ratio_pre:.4f}")

    def _write_debug_sample(
            self,
            artifacts_dir: Path,
            y_test: pd.Series,
            p_test_pre: np.ndarray,
            p_test_sigmoid: np.ndarray,
    ) -> None:
        try:
            dbg = pd.DataFrame({"y": y_test.to_numpy(), "pre": p_test_pre, "post": p_test_sigmoid})
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            dbg.sample(min(5000, len(dbg)), random_state=0).to_csv(artifacts_dir / "debug_test_preds_sample.csv",
                                                                   index=False)
        except Exception:
            pass

    def _compute_all_metrics(
            self,
            y_test: pd.Series,
            p_test_pre: np.ndarray,
            p_test_sigmoid: np.ndarray,
    ):
        y_test_np = y_test.to_numpy()
        computed_pre = self._compute_metrics_strict(y_test_np, p_test_pre)
        computed_post = self._compute_metrics_strict(y_test_np, p_test_sigmoid)
        return computed_pre, computed_post

    def _generate_all_plots(
            self,
            y_test: pd.Series,
            p_test_pre: np.ndarray,
            p_test_sigmoid: np.ndarray,
            artifacts_dir: Path,
    ):
        y_test_np = y_test.to_numpy()
        figs_curves = self.plot_roc_pr(y_test_np, p_test_pre, p_test_sigmoid, artifacts_dir)
        figs_reliab = self.plot_reliability_pair(y_test_np, p_test_pre, p_test_sigmoid, artifacts_dir)
        return figs_curves, figs_reliab

    def _merge_and_save_summary(
            self,
            metrics: Dict[str, Any],
            computed_pre: Dict[str, float],
            computed_post: Dict[str, float],
            figs_curves: Dict[str, str],
            figs_reliab: Dict[str, str],
            metrics_path: Path,
    ) -> None:
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


if __name__ == "__main__":
    OfflineEvaluator().main()
