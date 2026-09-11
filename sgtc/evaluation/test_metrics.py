import numpy as np

from sgtc.evaluation.metrics import lcc, srcc, rmse, utterance_metrics


def test_metrics_synthetic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.1, 1.9, 3.2, 3.8])
    # monotone (not affine): SRCC is exactly 1, LCC must match numpy reference
    assert abs(srcc(y, p) - 1.0) < 1e-12
    assert abs(lcc(y, p) - float(np.corrcoef(y, p)[0, 1])) < 1e-9
    assert abs(rmse(y, p) - np.sqrt(np.mean((y - p) ** 2))) < 1e-9
    # perfectly linear: LCC also exactly 1
    assert abs(lcc(y, 2.0 * y + 1.0) - 1.0) < 1e-9


def test_utterance_metrics_shape():
    m = utterance_metrics([3.1, 4.0, 1.2], [3.0, 4.2, 1.0])
    assert set(m) == {"lcc", "srcc", "rmse"}
    assert 0 <= abs(m["lcc"]) <= 1
