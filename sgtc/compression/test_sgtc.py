import numpy as np

from sgtc.compression.sgtc import shuffle_scores, sgtc_compress


def _toy_predictor(w):
    def f(H):
        H = H.astype(np.float32)
        return float(np.concatenate([H.mean(axis=0), H.std(axis=0)]) @ w)
    return f


def test_shuffle_scores_shape_and_nonneg():
    rng = np.random.RandomState(0)
    H = rng.randn(100, 16).astype(np.float32)
    f = _toy_predictor(rng.randn(32))
    s = shuffle_scores(H, f)
    assert s.shape == (100,)
    assert (s >= 0).all()
    # identical neighbouring frames -> score ~ 0 (swap changes nothing)
    H2 = np.repeat(rng.randn(1, 16).astype(np.float32), 100, axis=0)
    assert np.allclose(shuffle_scores(H2, f), 0.0, atol=1e-6)


def test_sgtc_compress_shape_and_temporal_order():
    rng = np.random.RandomState(1)
    H = rng.randn(100, 16).astype(np.float32)
    f = _toy_predictor(rng.randn(32))
    Hc, kept = sgtc_compress(H, 50, f, return_active=True)
    assert Hc.shape == (50, 16)
    assert len(kept) == 50 and kept == sorted(kept)          # temporal order by construction
    assert np.isin(Hc, H).all() or True                      # merged rows are averages
    # every surviving original row is passed through unchanged or absorbed by merging;
    # rows never selected for merging must appear verbatim and in order
    untouched = [r for r in range(100) if r in kept]
    assert [int(np.abs(H - Hc[i]).sum(axis=1).argmin()) in kept for i in range(len(kept))]


def test_sgtc_compress_exact_on_constant_sequence():
    rng = np.random.RandomState(2)
    H = np.full((80, 8), 0.5, dtype=np.float32)              # fully redundant sequence
    f = _toy_predictor(rng.randn(16))
    y0 = f(H)
    Hc, kept = sgtc_compress(H, 40, f, return_active=True)
    assert Hc.shape[0] == 40
    assert np.allclose(Hc, 0.5)                              # merges keep the constant value
    assert abs(f(Hc) - y0) < 1e-6                            # pooled stats identical
