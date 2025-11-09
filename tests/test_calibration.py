import json
from pathlib import Path


def test_calibration_losses_not_worse() -> None:
    p = Path("models/metrics_offline.json")
    assert p.exists(), "Run training before tests."
    d = json.loads(p.read_text())
    pre = d.get("metrics_pre", {})
    post = d.get("metrics_post", {})
    assert post.get("brier", 1.0) <= pre.get("brier", 1.0)
    assert post.get("nll", 1.0) <= pre.get("nll", 1.0)
