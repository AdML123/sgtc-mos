"""ToMe baseline: pairwise bipartite merging (adapted from Bolya et al. ICLR'3).

Alternating a/b groups -> each a finds its most-similar b (cosine) -> the
highest-similarity pair merges (mean) per round until T_target reached.
Faithful pairwise simplification of ToMe's bipartite soft matching (plan 4.2).
"""
import numpy as np


def tome_compress(H: np.ndarray, T_target: int) -> np.ndarray:
    H_work = H.astype(np.float32).copy()
    while H_work.shape[0] > T_target:
        T = H_work.shape[0]
        a_idx = np.arange(0, T, 2)
        b_idx = np.arange(1, T, 2)
        if len(b_idx) < len(a_idx):
            a_idx = a_idx[: len(b_idx)]
        a = H_work[a_idx]
        b = H_work[b_idx]
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        sim = a_n @ b_n.T                                  # (len(a), len(b))
        best_b = sim.argmax(axis=1)
        scores = sim[np.arange(len(a)), best_b]
        i = int(scores.argmax())                           # single best pair per round
        a_pos, b_pos = a_idx[i], b_idx[best_b[i]]
        H_work[b_pos] = (H_work[a_pos] + H_work[b_pos]) / 2.0
        H_work = np.delete(H_work, a_pos, axis=0)
    return H_work
