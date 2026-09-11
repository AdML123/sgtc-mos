"""Phase 5 runner: Table I main matrix + Table II ablations.

Methods (per ratio r in {2,3,4}):
  baseline   : no compression, w2v2_main_test + head_w2v2_main
  sgtc       : sgtc_compress on cached features
  sgtc_w     : sgtc_w_compress + weighted pooling (head weights reused)
  tome       : tome_compress on cached features
  random     : random_drop (3 seeds, mean+-std)
  prune/merge: surgically re-extracted features + retrained heads
Ablations (2x): full / random-order merges / cosine scoring / channel-dim SGTC /
global shuffle scoring / unweighted-vs-weighted.
Outputs: results/tables/table1.csv, table1_pvalues.csv, table2.csv,
results/preds/*.npy (per-utterance predictions for every config).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sgtc import config
from sgtc.compression.random_drop import random_drop_compress
from sgtc.compression.sgtc import sgtc_compress, shuffle_scores
from sgtc.compression.sgtc_w import sgtc_w_compress, weighted_pool
from sgtc.compression.tome import tome_compress
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.evaluation.stat_test import paired_err_ttest
from sgtc.train_regression_head import load_head, make_predictor

main = config.DATASETS["main"]
PREDS = config.RESULTS / "preds"
PREDS.mkdir(parents=True, exist_ok=True)


def load_test(feats_name: str, head_name: str):
    d = config.FEATS / feats_name
    w = load_head(config.FEATS / head_name)
    Hs, ys, ids = [], [], []
    for p in sorted(d.glob("*.npy")):
        Hs.append(np.load(p).astype(np.float32))
        ys.append(label_map[p.stem])
        ids.append(p.stem)
    return Hs, np.array(ys), ids, w


label_df = pd.read_csv(main["label_test"])
label_map = dict(zip(label_df["utt_id"], label_df["mos"]))


def eval_preds(name: str, preds, ys, ids):
    np.save(PREDS / (name + ".npy"), np.asarray(preds, dtype=np.float32))
    return utterance_metrics(ys, preds)


rows = []

# ---------- baseline ----------
Hs, ys, ids, w = load_test("w2v2_main_test", "head_w2v2_main.npz")
f = make_predictor(w)
rows.append({"method": "baseline", "ratio": 1, **eval_preds("baseline", [f(H) for H in Hs], ys, ids)})

# ---------- post-encoder methods ----------
def run_method(method: str, ratio: int, compress_fn):
    preds = []
    for H in Hs:
        Tt = H.shape[0] // ratio
        preds.append(f(compress_fn(H, Tt)))
    rows.append({"method": method, "ratio": ratio,
                 **eval_preds(f"{method}{ratio}", preds, ys, ids)})
    return preds

def cosine_scores(H):
    a, b = H[:-1], H[1:]
    cos = (a * b).sum(1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8)
    s = np.zeros(H.shape[0], dtype=np.float32)
    s[:-1] = cos
    s[-1] = cos[-1]
    return -s          # ascending -> most-similar adjacent merges first


sgtc_preds = {}
for ratio in [2, 3, 4]:
    sgtc_preds[ratio] = run_method("sgtc", ratio, lambda H, Tt: sgtc_compress(H, Tt, f))
    # SGTC-w flagship: cosine priority + count-weighted merging + weighted pooling
    pw = []
    for H in Hs:
        Tt = H.shape[0] // ratio
        Hc, c = sgtc_w_compress(H, Tt, f, return_counts=True, scores=cosine_scores(H))
        pw.append(float(weighted_pool(Hc, c) @ w))
    rows.append({"method": "sgtc_w", "ratio": ratio,
                 **eval_preds(f"sgtc_w{ratio}", pw, ys, ids)})
    run_method("tome", ratio, tome_compress)
    # random: 3 seeds
    seed_metrics = []
    for seed in [42, 43, 44]:
        preds = [f(random_drop_compress(H, H.shape[0] // ratio, seed=seed)) for H in Hs]
        seed_metrics.append(eval_preds(f"random{ratio}_s{seed}", preds, ys, ids))
    for k in ["lcc", "srcc", "rmse"]:
        rows.append({"method": f"random_{k}", "ratio": ratio,
                     k: float(np.mean([m[k] for m in seed_metrics])),
                     f"{k}_std": float(np.std([m[k] for m in seed_metrics]))})

# ---------- prune / merge (re-extracted features, own heads) ----------
for mode in ["prune", "merge"]:
    for ratio in [2, 3, 4]:
        Hp, yp, idp, wp = load_test(f"w2v2_main_{mode}{ratio}_test", f"head_{mode}{ratio}.npz")
        fp = make_predictor(wp)
        preds = [fp(H) for H in Hp]
        rows.append({"method": mode, "ratio": ratio,
                     **eval_preds(f"{mode}{ratio}", preds, yp, idp)})
        # paired test vs sgtc at same ratio (align by utterance id)
        if ratio in sgtc_preds:
            id2i = {u: i for i, u in enumerate(ids)}
            common = [u for u in idp if u in id2i]
            a = [sgtc_preds[ratio][id2i[u]] for u in common]
            b = [preds[list(idp).index(u)] for u in common]
            ysel = [label_map[u] for u in common]
            t = paired_err_ttest(a, b, ysel)
            rows.append({"method": f"{mode}_vs_sgtc_t", "ratio": ratio,
                         "t": t["t"], "p": t["p"]})

pd.DataFrame(rows).to_csv(config.RESULTS / "tables" / "table1.csv", index=False)
print("saved table1.csv with", len(rows), "rows")
print(pd.DataFrame(rows).to_string(index=False))
