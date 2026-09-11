"""B1 shuffle diagnosis (spec section 2, decision gate).

For each utterance feature H in R^{T x N}, compare MOS predictions under:
  original / time_shuffle (permute rows) / chan_shuffle (permute cols) /
  block_shuffle (permute blocks of `block` rows) / adjacent_swap (swap ~frac
  neighbouring pairs).  Utterance-level LCC/SRCC/RMSE per seed.
"""
import numpy as np

from sgtc.evaluation.metrics import utterance_metrics


def block_shuffle(H: np.ndarray, rng: np.random.RandomState, block: int = 50) -> np.ndarray:
    T = H.shape[0]
    n_blocks = T // block
    if n_blocks <= 1:
        return H
    H_trunc = H[: n_blocks * block]
    order = rng.permutation(n_blocks)
    return H_trunc.reshape(n_blocks, block, -1)[order].reshape(-1, H.shape[1])


def adjacent_swap(H: np.ndarray, rng: np.random.RandomState, frac: float = 0.1) -> np.ndarray:
    Hs = H.copy()
    T = H.shape[0]
    n_swap = max(1, int(T * frac) // 2)
    starts = rng.choice(T - 1, size=n_swap, replace=False)
    for t in starts:
        Hs[[t, t + 1]] = Hs[[t + 1, t]]
    return Hs


def shuffle_diagnosis(Hs, ys, f_predict, n_seeds: int = 5, block: int = 50) -> dict:
    Hs = [np.asarray(H, dtype=np.float32) for H in Hs]
    ops = {
        "original": lambda H, rng: H,
        "time_shuffle": lambda H, rng: H[rng.permutation(H.shape[0])],
        "chan_shuffle": lambda H, rng: H[:, rng.permutation(H.shape[1])],
        "block_shuffle": lambda H, rng: block_shuffle(H, rng, block),
        "adjacent_swap": lambda H, rng: adjacent_swap(H, rng),
    }
    out = {k: [] for k in ops}
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        for name, fn in ops.items():
            preds = [f_predict(fn(H, rng)) for H in Hs]
            out[name].append(utterance_metrics(list(ys), preds))
    return out
