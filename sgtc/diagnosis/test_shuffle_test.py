import numpy as np

from sgtc.diagnosis.shuffle_test import adjacent_swap, block_shuffle, shuffle_diagnosis


def _toy_predictor(w):
    def f(H):
        H = H.astype(np.float32)
        return float(np.concatenate([H.mean(axis=0), H.std(axis=0)]) @ w)
    return f


def test_shuffle_diagnosis_runs_and_shapes():
    rng = np.random.RandomState(0)
    Hs = [rng.randn(50, 8).astype(np.float32) for _ in range(20)]
    ys = rng.uniform(1.0, 5.0, size=20)
    f = _toy_predictor(rng.randn(16))
    res = shuffle_diagnosis(Hs, ys, f, n_seeds=2)
    assert set(res) == {"original", "time_shuffle", "chan_shuffle", "block_shuffle", "adjacent_swap"}
    for k, seeds in res.items():
        assert len(seeds) == 2
        for d in seeds:
            assert set(d) == {"lcc", "srcc", "rmse"}


def test_stat_pooling_predictor_is_permutation_invariant():
    rng = np.random.RandomState(1)
    H = rng.randn(60, 8).astype(np.float32)
    f = _toy_predictor(rng.randn(16))
    H_time = H[rng.permutation(H.shape[0])]
    H_chan = H[:, rng.permutation(H.shape[1])]
    assert abs(f(H) - f(H_time)) < 1e-5     # time order irrelevant
    assert abs(f(H) - f(H_chan)) > 1e-3     # channel order matters (w nonzero)


def test_block_shuffle_keeps_blocks_intact():
    rng = np.random.RandomState(2)
    H = np.arange(100 * 2, dtype=np.float32).reshape(100, 2)
    Hs = block_shuffle(H, rng, block=50)
    # each row itself must be preserved somewhere (blocks moved as wholes)
    rows = {tuple(r) for r in H}
    assert all(tuple(r) in rows for r in Hs)


def test_adjacent_swap_changes_few_positions():
    rng = np.random.RandomState(3)
    H = np.arange(200 * 2, dtype=np.float32).reshape(200, 2)
    Hs = adjacent_swap(H, rng, frac=0.1)
    diff = (Hs != H).any(axis=1).sum()
    assert 0 < diff <= 40  # ~10% of frames swapped (each swap touches 2 rows)
