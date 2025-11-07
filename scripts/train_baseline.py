import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pathlib import Path
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import calibration_curve


class BaselineTrainer:
    """Train baseline classifiers, calibrate probabilities, and export artifacts.

    Steps:
    1) Load dataset (CSV or synthetic fallback) and sort by Time.
    2) Temporal split: train / valid / calibration / test.
    3) Build preprocessing + baseline models; select on valid (AP).
    4) Refit on train+valid; calibrate on calibration split (sigmoid, isotonic).
    5) Evaluate on test (AUC, AP, Brier, NLL); save reliability diagrams.
    6) Persist calibrated model and metrics.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    # ---------------------------- Data ----------------------------
    def load_dataset(self, path: str | None = None, sample_frac: float | None = None) -> pd.DataFrame:
        """Load Credit Card Fraud dataset.

        Preference order:
        1) Local CSV at data/creditcard.csv
        2) Remote via KaggleHub (mlg-ulb/creditcardfraud → creditcard.csv)

        """
        if path is None:
            path = "data/creditcard.csv"

        csv_path = Path(path)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            try:
                import kagglehub  # type: ignore
                from kagglehub import KaggleDatasetAdapter  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Dataset not found locally and KaggleHub is unavailable. "
                    "Install with: pip install kagglehub[pandas-datasets]"
                ) from exc

            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "mlg-ulb/creditcardfraud",
                "creditcard.csv",
            )

        if "Time" not in df.columns or "Class" not in df.columns:
            raise ValueError("Dataset must contain 'Time' and 'Class' columns.")

        if sample_frac is not None and 0 < sample_frac < 1:
            df = df.sample(frac=sample_frac, random_state=self.random_state).sort_values("Time")

        return df.sort_values("Time").reset_index(drop=True)

    def temporal_split(
            self,
            df: pd.DataFrame,
            ratios: Tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2),
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Temporal split into train, valid, calibration, test.

        Splits strictly by increasing Time. Drops Time from features.
        """
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("Ratios must sum to 1.0")

        df_sorted = df.sort_values("Time").reset_index(drop=True)
        n = len(df_sorted)
        n_train = int(n * ratios[0])
        n_valid = int(n * ratios[1])
        n_calib = int(n * ratios[2])
        n_test = n - n_train - n_valid - n_calib
        if min(n_train, n_valid, n_calib, n_test) <= 0:
            raise ValueError("Temporal split produced an empty segment; reduce ratios or increase data.")

        parts = np.cumsum([n_train, n_valid, n_calib])
        df_train = df_sorted.iloc[: parts[0]]
        df_valid = df_sorted.iloc[parts[0]: parts[1]]
        df_calib = df_sorted.iloc[parts[1]: parts[2]]
        df_test = df_sorted.iloc[parts[2]:]

        feature_cols = [c for c in df.columns if c not in {"Class", "Time"}]

        X_train, y_train = df_train[feature_cols], df_train["Class"].astype(int)
        X_valid, y_valid = df_valid[feature_cols], df_valid["Class"].astype(int)
        X_calib, y_calib = df_calib[feature_cols], df_calib["Class"].astype(int)
        X_test, y_test = df_test[feature_cols], df_test["Class"].astype(int)
        return X_train, y_train, X_valid, y_valid, X_calib, y_calib, X_test, y_test

    # ------------------------ Modeling ------------------------
    def build_preprocessor(self, numeric_cols: List[str]) -> ColumnTransformer:
        """Numeric imputation + scaling on provided columns."""
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=True, with_std=True)),
            ]
        )
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
            ],
            remainder="drop",
        )
        return pre

    def candidate_models(self, pre: ColumnTransformer) -> List[Tuple[str, Dict[str, Any]]]:
        """Return a small set of model configs to try on the valid split."""
        models: List[Tuple[str, Dict[str, Any]]] = []

        # Logistic Regression variants
        for C in [0.1, 1.0, 10.0]:
            models.append(
                ("logreg", {"pipeline":
                                Pipeline(steps=[("pre", pre), ("clf",
                                                               LogisticRegression(C=C, penalty="l2",
                                                                                  solver="lbfgs", max_iter=2000,
                                                                                  class_weight="balanced",
                                                                                  random_state=self.random_state)
                                                               ),
                                                ]),
                            "label": f"LR(C={C})",
                            }
                 ))

        # Gradient Boosting (light grid)
        for n_estimators in [100, 200]:
            for learning_rate in [0.05, 0.1]:
                models.append(
                    ("gbdt", {"pipeline":
                                  Pipeline(steps=[("pre", pre), ("clf",
                                                                 GradientBoostingClassifier(
                                                                     n_estimators=n_estimators,
                                                                     learning_rate=learning_rate,
                                                                     max_depth=3,
                                                                     random_state=self.random_state)
                                                                 ),
                                                  ]),
                              "label": f"GBDT(n={n_estimators},lr={learning_rate})",
                              },
                     ))

        return models

    def select_base_model(
            self,
            candidates: List[Tuple[str, Dict[str, Any]]],
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_valid: pd.DataFrame,
            y_valid: pd.Series,
    ) -> Tuple[str, Pipeline, Dict[str, Any]]:
        """Fit each candidate on train, evaluate AP on valid, and select the best."""
        best_label = ""
        best_pipe: Pipeline | None = None
        best_score = -np.inf
        best_meta: Dict[str, Any] = {}

        for name, cfg in candidates:
            pipe: Pipeline = cfg["pipeline"]
            label: str = cfg["label"]
            pipe.fit(X_train, y_train)
            prob_valid = pipe.predict_proba(X_valid)[:, 1]
            ap = average_precision_score(y_valid, prob_valid)
            if ap > best_score:
                best_score = ap
                best_label = label
                best_pipe = pipe
                best_meta = {"model_name": name, "label": label, "valid_ap": float(ap)}

        if best_pipe is None:
            raise RuntimeError("No candidate model could be selected.")

        return best_label, best_pipe, best_meta

    # ---------------------- Calibration ----------------------
    def calibrate(
            self,
            base_prefit: Pipeline,
            X_cal: pd.DataFrame,
            y_cal: pd.Series,
            methods: Tuple[str, str] = ("sigmoid", "isotonic"),
    ) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
        """Fit calibrators on calibration split and select best by NLL then Brier."""
        results: List[Tuple[str, CalibratedClassifierCV, float, float]] = []
        for method in methods:
            calib = CalibratedClassifierCV(base_estimator=base_prefit, method=method, cv="prefit")
            calib.fit(X_cal, y_cal)
            p_cal = calib.predict_proba(X_cal)[:, 1]
            nll = log_loss(y_cal, p_cal, labels=[0, 1])
            brier = brier_score_loss(y_cal, p_cal)
            results.append((method, calib, nll, brier))

        # Sort by NLL then Brier
        results.sort(key=lambda t: (t[2], t[3]))
        best_method, best_model, best_nll, best_brier = results[0]
        meta = {"calibration_method": best_method, "calib_nll": float(best_nll), "calib_brier": float(best_brier)}
        return best_model, meta

    # ----------------------- Evaluation -----------------------
    def evaluate_offline(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
        """Compute ROC-AUC, PR-AUC, Brier, NLL."""
        metrics: Dict[str, float] = {}
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["brier"] = float(brier_score_loss(y_true, y_score))
        metrics["nll"] = float(log_loss(y_true, y_score, labels=[0, 1]))
        return metrics

    def _save_reliability_plot(self, y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str) -> None:
        """Save a reliability diagram to the given path."""
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=15, strategy="quantile")
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.xlabel("Predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()

    # ------------------------- Orchestration -------------------------
    def main(
            self,
            data_path: str | None = None,
            models_dir: str = "models",
            artifacts_dir: str = "artifacts",
            split_ratios: Tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2),
            sample_frac: float | None = None,
    ) -> None:
        # Load
        df = self.load_dataset(path=data_path, sample_frac=sample_frac)

        # Split
        X_train, y_train, X_valid, y_valid, X_cal, y_cal, X_test, y_test = self.temporal_split(df, ratios=split_ratios)

        # Preprocessor and candidates
        numeric_cols = list(X_train.columns)
        pre = self.build_preprocessor(numeric_cols)
        candidates = self.candidate_models(pre)

        # Select base on valid
        best_label, base_model, base_meta = self.select_base_model(candidates, X_train, y_train, X_valid, y_valid)

        # Refit base on train+valid
        X_trv = pd.concat([X_train, X_valid], axis=0)
        y_trv = pd.concat([y_train, y_valid], axis=0)
        base_model.fit(X_trv, y_trv)

        # Raw scores on test (pre-calibration)
        p_test_raw = base_model.predict_proba(X_test)[:, 1]
        metrics_raw = self.evaluate_offline(y_test.to_numpy(), p_test_raw)

        # Calibrate on calibration split and evaluate
        calib_model, calib_meta = self.calibrate(base_model, X_cal, y_cal)
        p_test_cal = calib_model.predict_proba(X_test)[:, 1]
        metrics_cal = self.evaluate_offline(y_test.to_numpy(), p_test_cal)

        # Save reliability diagrams
        artifacts_path = Path(artifacts_dir)
        self._save_reliability_plot(y_test.to_numpy(), p_test_raw, artifacts_path / "reliability_pre.png",
                                    title=f"Reliability (pre) - {best_label}")
        self._save_reliability_plot(y_test.to_numpy(), p_test_cal, artifacts_path / "reliability_post.png",
                                    title=f"Reliability (post) - {calib_meta['calibration_method']}")

        # Persist model and metrics
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        model_out = models_path / "model.joblib"
        dump(calib_model, model_out)

        metrics = {
            "base_model": base_meta,
            "calibration": calib_meta,
            "metrics_pre": metrics_raw,
            "metrics_post": metrics_cal,
            "artifacts": {
                "reliability_pre": str((artifacts_path / "reliability_pre.png").as_posix()),
                "reliability_post": str((artifacts_path / "reliability_post.png").as_posix()),
            },
        }
        with open(models_path / "metrics_offline.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Console summary
        print("Selected base:", base_meta)
        print("Calibration:", calib_meta)
        print("Pre-calibration metrics:", metrics_raw)
        print("Post-calibration metrics:", metrics_cal)


if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.main()
