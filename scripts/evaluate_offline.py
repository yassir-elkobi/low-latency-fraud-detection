from typing import Any, Dict


class OfflineEvaluator:
    """Offline evaluation utilities and figure generation for the report.

    Generates ROC and PR curves, score histograms, and reliability diagrams
    before and after calibration, and compiles a metrics table for the report.
    """

    def compute_metrics(self, y_true: Any, y_score: Any) -> Dict[str, float]:
        """Return AUC, AP, Brier, and NLL for given labels and scores."""
        pass

    def plot_curves(self, y_true: Any, y_score: Any, out_dir: str) -> None:
        """Create ROC and PR curve figures and save them to disk."""
        pass

    def plot_reliability(self, y_true: Any, y_score_raw: Any, y_score_cal: Any, out_dir: str) -> None:
        """Generate pre/post calibration reliability diagrams."""
        pass

    def export_table(self, metrics_pre: Dict[str, float], metrics_post: Dict[str, float], out_path: str) -> None:
        """Save a compact CSV/JSON table summarizing key metrics."""
        pass


"""
Offline evaluation and figure generation skeleton.

Produces ROC/PR curves, reliability diagrams before/after calibration,
histograms of scores, and summary tables for the report.
"""

from typing import Any, Dict


class OfflineEvaluator:
    """Encapsulates plotting and table generation for offline evaluation."""

    def plot_roc_pr(self, y_true: Any, y_score: Any, out_dir: str) -> None:
        """Generate ROC and PR curves and save figures."""
        pass

    def plot_reliability(self, y_true: Any, y_score_raw: Any, y_score_cal: Any, out_dir: str) -> None:
        """Generate pre/post-calibration reliability diagrams."""
        pass

    def export_table(self, metrics_before: Dict[str, float], metrics_after: Dict[str, float], out_path: str) -> None:
        """Write a compact metrics comparison table to disk."""
        pass
