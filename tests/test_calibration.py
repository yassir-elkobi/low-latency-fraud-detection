class TestCalibration:
    """Calibration tests to ensure post-calibration improvements.

    Checks that Brier score and negative log-likelihood do not regress after
    calibration on a dedicated split, within a seeded and reproducible setup.
    """

    def test_brier_improves(self) -> None:
        """Brier score after calibration should be less than or equal to before."""
        pass

    def test_nll_improves(self) -> None:
        """NLL after calibration should be less than or equal to before."""
        pass


"""
Calibration tests skeleton.

Verifies that calibration improves Brier/NLL on a held-out calibration
split with fixed seeds (non-regression behavior).
"""


def test_calibration_improves_loss_skeleton() -> None:
    """Placeholder for calibration non-regression test."""
    pass
