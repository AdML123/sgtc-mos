"""Self-contained demo smoke test (runs on data/demo only, no model download).

Trains the ridge head on 10 demo train utterances, evaluates on 10 test
utterances, runs the shuffle diagnosis and the temporal compression comparison
on cached features, and compares every number against
results/demo_reference.json. Exit code 0 on PASS.

Run:  python -m sgtc.run_demo [--update-reference]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sgtc.compression.random_drop import random_drop_compress
from sgtc.compression.sgtc import sgtc_compress
from sgtc.compression.sgtc_w import sgtc_w_compress, weighted_pool
from sgtc.compression.tome import tome_compress
from sgtc.diagnosis.shuffle_test import shuffle_diagnosis
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.train_regression_head import pool

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "data" / "demo"
REF = ROOT / "results" / "demo_reference.json"
TOL = 1e-6


def load(split):
    df = pd.read_csv(DEMO / (split + "_mos.csv"))
    Hs, ys = [], []
    for _, r in df.iterrows():
        p = DEMO / "features" / (r["utt_id"] + ".npy")
        Hs.append(np.load(p).astype(np.float32))
        ys.append(r["mos"])
    return Hs, np.array(ys), df


def main():
    Hs_tr, ys_tr, _ = load("train")
    Hs_te, ys_te, _ = load("test")

    # ridge head on pooled statistics (same closed form as the paper)
    X = np.stack([pool(H) for H in Hs_tr])
    w = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ ys_tr)

    def f(H):
        return float(pool(H) @ w)

    out = {"baseline": utterance_metrics(ys_te, [f(H) for H in Hs_te])}

    diag = shuffle_diagnosis(Hs_te, ys_te, f, n_seeds=3)
    out["diag_d_time"] = float(np.mean([d["lcc"] for d in diag["time_shuffle"]]))
    out["diag_d_chan"] = out["baseline"]["lcc"] - float(
        np.mean([d["lcc"] for d in diag["chan_shuffle"]]))

    preds_w, preds_u, preds_t, preds_r = [], [], [], []
    for H in Hs_te:
        Tt = H.shape[0] // 2
        preds_u.append(f(sgtc_compress(H, Tt, f)))
        Hc, c = sgtc_w_compress(H, Tt, f, return_counts=True)
        preds_w.append(float(weighted_pool(Hc, c) @ w))
        preds_t.append(f(tome_compress(H, Tt)))
        preds_r.append(f(random_drop_compress(H, Tt, seed=42)))
    out["sgtc_2x"] = utterance_metrics(ys_te, preds_u)
    out["sgtc_w_2x"] = utterance_metrics(ys_te, preds_w)
    out["tome_2x"] = utterance_metrics(ys_te, preds_t)
    out["random_2x"] = utterance_metrics(ys_te, preds_r)

    flat = json.dumps(out, sort_keys=True)
    if "--update-reference" in sys.argv:
        REF.write_text(json.dumps(out, indent=2, sort_keys=True))
        print("reference updated ->", REF)
        return

    ref = json.loads(REF.read_text())
    bad = []
    for k, v in _flatten(out).items():
        r = _flatten(ref).get(k)
        if r is None or abs(v - r) > TOL:
            bad.append((k, v, r))
    if bad:
        for k, v, r in bad:
            print(f"MISMATCH {k}: got {v:.6f}, reference {r}")
        sys.exit(1)
    print("DEMO PASS: pipeline mechanics verified against "
          "results/demo_reference.json "
          f"(tolerance {TOL}). NOTE: this 10-utterance subset is a smoke "
          "test -- the absolute numbers are noise-level by design; the "
          "paper's numbers live in results/tables/ and require the full "
          "corpus (see README).")


def _flatten(d, prefix=""):
    flat = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flatten(v, prefix + k + "."))
        else:
            flat[prefix + k] = v
    return flat


if __name__ == "__main__":
    main()
