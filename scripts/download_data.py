from typing import Optional


class DataDownloader:
    """Dataset acquisition and CI-friendly sampling utility.

    Downloads the Credit Card Fraud dataset if available or builds a small,
    reproducible stratified sample for CI runs when the full dataset is absent.
    """

    def fetch(self, dest_path: str) -> None:
        """Download the dataset to the specified local path."""
        pass

    def create_sample(self, src_path: Optional[str], dest_path: str, frac: float, seed: int) -> None:
        """Create a small stratified sample for fast CI workflows."""
        pass


"""
Dataset acquisition and CI sample generation skeleton.

Downloads the Credit Card Fraud dataset if available, or creates a small,
stratified sample for CI purposes to keep pipelines fast and reproducible.
"""

from typing import Any


class DataDownloader:
    """Encapsulates dataset fetching and CI-friendly sampling utilities."""

    def fetch(self, destination: str) -> None:
        """Download the dataset to the specified destination path."""
        pass

    def make_ci_sample(self, source_path: str, out_path: str, frac: float, seed: int = 42) -> None:
        """Create a small stratified sample for continuous integration runs."""
        pass
