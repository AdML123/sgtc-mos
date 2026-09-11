"""SGTC core: shuffle-guided temporal compression (paper Eqs. 1-2, spec 3.2/3.3).

Eq.(1) sensitivity: s_t = mean over neighbours t' in {t-1, t+1} (clipped) of
|f(H) - f(swap(H, t, t'))|.  Eq.(2) merge: lowest-sensitivity frames merge
first into their cosine-nearest active neighbour (mean merge); output keeps the
original temporal order.  No trainable parameters.
"""
import numpy as np


def shuffle_scores(H: np.ndarray, f_predict) -> np.ndarray:
    """Eq. (1): frame-level shuffle sensitivity."""
    T = H.shape[0]
    y0 = f_predict(H)
    s = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if T == 1:
            break
        nbs = ([t + 1] if t == 0 else [t - 1] if t == T - 1 else [t - 1, t + 1])
        tot = 0.0
        for nb in nbs:
            Hs = H.copy()
            Hs[[t, nb]] = Hs[[nb, t]]
            tot += abs(y0 - f_predict(Hs))
        s[t] = tot / len(nbs)
    return s


def sgtc_compress(H: np.ndarray, T_target: int, f_predict, return_active: bool = False,
                  scores: np.ndarray = None):
    """Merge lowest-sensitivity frames first until len == T_target.

    Returns H_compressed[, sorted_active]; `sorted_active` are the surviving
    original indices (ascending), so the output provably keeps temporal order.
    `scores`: optional external priority (e.g. ablation variants); lowest first.
    """
    T = H.shape[0]
    if T <= T_target:
        Hc = H.astype(np.float32)
        return (Hc, list(range(T))) if return_active else Hc
    if scores is None:
        scores = shuffle_scores(H, f_predict)
    order = list(np.argsort(scores, kind="stable"))   # least sensitive first
    H_work = H.astype(np.float32).copy()
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
        H_work[best] = (H_work[idx] + H_work[best]) / 2.0
        active.remove(idx)
    kept = sorted(active)
    Hc = H_work[kept]
    return (Hc, kept) if return_active else Hc
