import numpy as np

from sgtc.compression.sgtc_w import sgtc_w_compress, weighted_pool


def _toy_predictor(w):
    def f(H):
        H = H.astype(np.float32)
        return float(np.concatenate([H.mean(axis=0), H.std(axis=0)]) @ w)
    return f


def test_weighted_pool_matches_unweighted_when_counts_equal():
    rng = np.random.RandomState(0)
    H = rng.randn(50, 8).astype(np.float32)
    c = np.ones(50, dtype=np.float32)
    zp = weighted_pool(H, c)
    assert np.allclose(zp[:8], H.mean(axis=0), atol=1e-6)
    assert np.allclose(zp[8:], H.std(axis=0), atol=1e-5)


def test_prop1_mean_exact_std_bounded():
    """Prop.1 (v2): weighted MEAN preserved exactly for ANY input; weighted
    pooling beats unweighted pooling when the input has real redundancy
    (near-duplicate regions, as in speech features)."""
    rng = np.random.RandomState(1)
    H = rng.randn(120, 8).astype(np.float32)
    f = _toy_predictor(rng.randn(16))
    Hc, counts = sgtc_w_compress(H, 60, f, return_counts=True)
    assert Hc.shape[0] == 60 and counts.sum() == 120 and (counts >= 1).all()
    zp = weighted_pool(Hc, counts)
    assert np.allclose(zp[:8], H.mean(axis=0), atol=1e-4)          # mean exact, iid too

    # redundant regime: long near-duplicate block + few informative frames
    base = rng.randn(1, 8).astype(np.float32)
    redundant = np.repeat(base, 100, axis=0) + rng.randn(100, 8).astype(np.float32) * 1e-3
    informative = rng.randn(20, 8).astype(np.float32) * 3
    H2 = np.vstack([redundant, informative])
    H2c, c2 = sgtc_w_compress(H2, 60, f, return_counts=True)
    ref2 = np.concatenate([H2.mean(axis=0), H2.std(axis=0)])
    err_w = np.abs(weighted_pool(H2c, c2) - ref2).max()
    err_u = np.abs(np.concatenate([H2c.mean(axis=0), H2c.std(axis=0)]) - ref2).max()
    assert err_w < err_u, (err_w, err_u)                            # wins under redundancy


def test_prop1_exact_for_identical_merged_frames():
    """When all frames are identical, SGTC-w is exactly lossless at any ratio."""
    rng = np.random.RandomState(2)
    H = np.tile(rng.randn(1, 6).astype(np.float32), (90, 1))
    w = rng.randn(12)
    f = _toy_predictor(w)
    for target in (45, 30, 23):
        Hc, counts = sgtc_w_compress(H, target, f, return_counts=True)
        zp = weighted_pool(Hc, counts)
        assert np.allclose(zp, np.concatenate([H.mean(axis=0), H.std(axis=0)]), atol=1e-5)
        assert abs(float(zp @ w) - f(H)) < 1e-3


def test_counts_sum_preserved_across_ratios():
    rng = np.random.RandomState(2)
    H = rng.randn(90, 4).astype(np.float32)
    f = _toy_predictor(rng.randn(8))
    for target in (45, 30, 23):
        Hc, counts = sgtc_w_compress(H, target, f, return_counts=True)
        assert Hc.shape[0] == target and float(counts.sum()) == 90.0
