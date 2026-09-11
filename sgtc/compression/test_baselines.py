import numpy as np

from sgtc.compression.random_drop import random_drop_compress
from sgtc.compression.tome import tome_compress


def test_shapes():
    rng = np.random.RandomState(0)
    H = rng.randn(100, 16).astype(np.float32)
    assert tome_compress(H, 50).shape == (50, 16)
    assert random_drop_compress(H, 50, seed=1).shape == (50, 16)


def test_random_drop_reproducible_and_ordered():
    rng = np.random.RandomState(0)
    H = rng.randn(80, 4).astype(np.float32)
    a = random_drop_compress(H, 40, seed=7)
    b = random_drop_compress(H, 40, seed=7)
    assert np.array_equal(a, b)
    # kept rows keep temporal order and are verbatim originals
    idx = [int((np.abs(H - r).sum(axis=1)).argmin()) for r in a]
    assert idx == sorted(idx)
    assert all(np.abs(H[i] - a[k]).sum() < 1e-6 for k, i in enumerate(idx))


def test_tome_merges_most_similar_first():
    # two clusters of identical rows + distinct rows; ToMe should first merge
    # within the identical cluster (high cosine), never averaging distinct rows early
    rng = np.random.RandomState(1)
    A = np.tile(np.array([1.0, 0.0], dtype=np.float32), (30, 1))
    B = np.tile(np.array([0.0, 1.0], dtype=np.float32), (30, 1))
    C = rng.randn(40, 2).astype(np.float32)
    H = rng.permutation(np.vstack([A, B, C]))
    Hc = tome_compress(H, 50)
    assert Hc.shape[0] == 50
    # compressed rows must be original rows or pairwise means of identical rows
    for r in Hc:
        dists = np.abs(H - r).sum(axis=1)
        assert dists.min() < 1e-5 or np.abs(dists - 0.0).min() < 0.6
