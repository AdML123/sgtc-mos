"""SGTC-w: count-weighted variant (spec v3 3.4, paper Proposition 1).

Merge schedule of sgtc_compress, but merged frames use the COUNT-WEIGHTED
average and carry the number of originals they represent.  Guarantees:
  (a) the count-weighted pooled MEAN equals the original pooled mean exactly
      (any merge order, any ratio);
  (b) the weighted STD shrinks only by the dispersion of the merged pairs
      (exactly preserved when merged frames coincide) -- near-duplicate
      low-sensitivity merges keep this perturbation small.
Empirically near-lossless for stat-pooling scorers; see test_sgtc_w.py.
"""
import numpy as np

from sgtc.compression.sgtc import shuffle_scores


def weighted_pool(H: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Weighted counterpart of train_regression_head.pool."""
    H = H.astype(np.float32)
    c = np.asarray(counts, dtype=np.float32)
    mu = (c[:, None] * H).sum(axis=0) / c.sum()
    var = (c[:, None] * (H - mu) ** 2).sum(axis=0) / c.sum()
    return np.concatenate([mu, np.sqrt(np.maximum(var, 0.0))])


def sgtc_w_compress(H: np.ndarray, T_target: int, f_predict, return_counts: bool = False,
                    scores: np.ndarray = None):
    """Merge schedule of sgtc_compress with count-weighted merging (Eq.2').

    `scores`: optional external merge priority (lowest merges first).
    """
    T = H.shape[0]
    if T <= T_target:
        Hc = H.astype(np.float32)
        return (Hc, np.ones(T, dtype=np.float32)) if return_counts else Hc
    if scores is None:
        scores = shuffle_scores(H, f_predict)
    order = list(np.argsort(scores, kind="stable"))
    H_work = H.astype(np.float32).copy()
    counts = np.ones(T, dtype=np.float32)
    active = list(range(T))
    for idx in order:
        if len(active) <= T_target:
            break
        if idx not in active:
            continue
        pos = active.index(idx)
        cands = ([active[pos - 1]] if pos > 0 else []) + \
                ([active[pos + 1]] if pos < len(active) - 1 else [])
        if not cands:
            continue
        hi = H_work[idx]
        ni = np.linalg.norm(hi) + 1e-8
        best = max(cands, key=lambda c: float(hi @ H_work[c] /
                                              (ni * (np.linalg.norm(H_work[c]) + 1e-8))))
        ci, cb = counts[idx], counts[best]
        H_work[best] = (ci * H_work[idx] + cb * H_work[best]) / (ci + cb)
        counts[best] = ci + cb
        active.remove(idx)
    kept = sorted(active)
    Hc = H_work[kept]
    return (Hc, counts[kept]) if return_counts else Hc
