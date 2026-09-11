"""Paired significance tests between two methods' per-utterance predictions.

We compare per-utterance absolute errors |pred - mos| with a paired t-test
(difference in accuracy, not difference in raw predictions), plus utterance
level bootstrap for the LCC gap.
"""
import numpy as np
from scipy import stats


def paired_err_ttest(pred_a, pred_b, ys) -> dict:
    """H1: |err_a| != |err_b| (two-sided) on paired utterances."""
    ea = np.abs(np.asarray(pred_a) - np.asarray(ys))
    eb = np.abs(np.asarray(pred_b) - np.asarray(ys))
    t, p = stats.ttest_rel(ea, eb)
    return {"t": float(t), "p": float(p)}


def lcc_gap_bootstrap(pred_a, pred_b, ys, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI for LCC(a) - LCC(b) at utterance level."""
    from sgtc.evaluation.metrics import lcc
    ys = np.asarray(ys)
    rng = np.random.RandomState(seed)
    gaps = []
    n = len(ys)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(ys[idx])) < 2:
            continue
        gaps.append(lcc(ys[idx], np.asarray(pred_a)[idx]) - lcc(ys[idx], np.asarray(pred_b)[idx]))
    gaps = np.array(gaps)
    return {"mean": float(gaps.mean()), "ci95": [float(np.percentile(gaps, 2.5)),
                                                 float(np.percentile(gaps, 97.5))]}
