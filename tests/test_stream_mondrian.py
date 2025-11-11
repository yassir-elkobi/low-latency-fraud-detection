import numpy as np
from scripts.simulate_stream import StreamSimulator


def test_mondrian_borderline_batch_improves_fraud_and_sets() -> None:
    n_pairs = 200
    probs_pos = np.full(n_pairs, 0.55, dtype=float)
    probs_neg = np.full(n_pairs, 0.45, dtype=float)
    # Interleave to simulate stream arrival
    probs = np.empty(n_pairs * 2, dtype=float)
    probs[0::2] = probs_pos
    probs[1::2] = probs_neg
    y = np.empty_like(probs, dtype=int)
    y[0::2] = 1
    y[1::2] = 0

    sim = StreamSimulator(alpha=0.05, mode="window", window=200, decay=0.01, label_delay=1, warmup=0)
    log = sim.simulate(probs=probs, y_true=y)
    assert log.get("avg_set_size") is not None and float(log["avg_set_size"]) > 1.0
    cov_pos = log.get("final_coverage_pos")
    assert cov_pos is not None and float(cov_pos) >= 0.8
