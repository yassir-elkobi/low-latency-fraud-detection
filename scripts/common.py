from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load Credit Card Fraud dataset from local CSV only and sort by Time."""
    if path is None:
        path = "data/creditcard.csv"
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}.")
    df = pd.read_csv(csv_path)
    if "Time" not in df.columns or "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Time' and 'Class' columns.")
    return df.sort_values("Time").reset_index(drop=True)


def temporal_split(
        df: pd.DataFrame,
        ratios: Tuple[float, float, float, float] = (0.6, 0.1, 0.1, 0.2),
):
    """Temporal split train/valid/calibration/test by increasing Time."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Ratios must sum to 1.0")
    n = len(df)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    n_calib = int(n * ratios[2])
    parts = np.cumsum([n_train, n_valid, n_calib])
    df_train = df.iloc[: parts[0]]
    df_valid = df.iloc[parts[0]: parts[1]]
    df_calib = df.iloc[parts[1]: parts[2]]
    df_test = df.iloc[parts[2]:]
    feature_cols = [c for c in df.columns if c not in {"Class", "Time"}]
    return (
        df_train[feature_cols], df_train["Class"].astype(int),
        df_valid[feature_cols], df_valid["Class"].astype(int),
        df_calib[feature_cols], df_calib["Class"].astype(int),
        df_test[feature_cols], df_test["Class"].astype(int),
    )


def build_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    """Numeric imputation + scaling."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric_pipeline, numeric_cols)], remainder="drop")


def compute_metrics(y_true, y_score) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC, Brier, NLL."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "nll": float(log_loss(y_true, y_score, labels=[0, 1])),
    }
