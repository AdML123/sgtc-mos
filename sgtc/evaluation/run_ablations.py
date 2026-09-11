"""Phase 5b: Table II ablations (2x compression) + global-shuffle scores.

Variants share SGTC's merge skeleton, only the priority differs:
  full      : Eq.(1) local swap sensitivity (reference row, from Table I)
  random    : random priority
  cosine    : adjacent cosine similarity (geometry, not task)
  attention : first/last-layer attention received (GPU cache -> scores.npy)
  global    : Eq.(1) with far/random partner instead of neighbour
  sgtc_w    : count-weighted merge + weighted pooling (Table I row)
Channel-dim SGTC (channel selection + head retrain) lives in run_channel_sgtc.py.
"""
import json

import numpy as np
import pandas as pd

from sgtc import config
from sgtc.compression.sgtc import sgtc_compress
from sgtc.compression.sgtc_w import sgtc_w_compress, weighted_pool
from sgtc.compression.tome import tome_compress
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.train_regression_head import load_head, make_predictor

main = config.DATASETS["main"]
label_df = pd.read_csv(main["label_test"])
label_map = dict(zip(label_df["utt_id"], label_df["mos"]))

Hs, ys, ids = [], [], []
for p in sorted((config.FEATS / "w2v2_main_test").glob("*.npy")):
    Hs.append(np.load(p).astype(np.float32))
    ys.append(label_map[p.stem])
    ids.append(p.stem)
ys = np.array(ys)
w = load_head(config.FEATS / "head_w2v2_main.npz")
f = make_predictor(w)


def eval_variant(name: str, preds):
    m = utterance_metrics(ys, preds)
    np.save(config.RESULTS / "preds" / ("abl_" + name + ".npy"), np.asarray(preds, np.float32))
    print(f"{name:12s} LCC={m['lcc']:.4f} SRCC={m['srcc']:.4f} RMSE={m['rmse']:.4f}", flush=True)
    return {"variant": name, **m}


rows = []
ratio = 2

# --- reference: full SGTC (Eq.1) ---
preds = [f(sgtc_compress(H, H.shape[0] // ratio, f)) for H in Hs]
rows.append(eval_variant("full", preds))

# --- random priority ---
rng = np.random.RandomState(42)
preds = [f(sgtc_compress(H, H.shape[0] // ratio, f,
                         scores=rng.permutation(H.shape[0]).astype(np.float32))) for H in Hs]
rows.append(eval_variant("random", preds))

# --- cosine (adjacent similarity priority): merge most-similar-neighbour first ---
def cosine_scores(H):
    a, b = H[:-1], H[1:]
    cos = (a * b).sum(1) / ((np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-8)
    s = np.zeros(H.shape[0], dtype=np.float32)
    s[:-1] = cos          # high similarity == low sensitivity proxy
    s[-1] = cos[-1]
    return -s             # ascending order merges most-similar first

preds = [f(sgtc_compress(H, H.shape[0] // ratio, f, scores=cosine_scores(H))) for H in Hs]
rows.append(eval_variant("cosine", preds))

# --- global shuffle sensitivity: partner = random far position ---
def global_scores(H, f_pred, seed=0):
    T = H.shape[0]
    r = np.random.RandomState(seed)
    y0 = f_pred(H)
    s = np.zeros(T, dtype=np.float32)
    for t in range(T):
        partner = int(r.randint(0, T))
        while partner == t:
            partner = int(r.randint(0, T))
        Hs_ = H.copy()
        Hs_[[t, partner]] = Hs_[[partner, t]]
        s[t] = abs(y0 - f_pred(Hs_))
    return s

preds = [f(sgtc_compress(H, H.shape[0] // ratio, f, scores=global_scores(H, f))) for H in Hs]
rows.append(eval_variant("global", preds))

# --- ToMe (geometry baseline for reference in this table) ---
preds = [f(tome_compress(H, H.shape[0] // ratio)) for H in Hs]
rows.append(eval_variant("tome", preds))

# --- SGTC-w (Eq.1 priority) ---
preds = []
for H in Hs:
    Hc, c = sgtc_w_compress(H, H.shape[0] // ratio, f, return_counts=True)
    preds.append(float(weighted_pool(Hc, c) @ w))
rows.append(eval_variant("sgtc_w", preds))

# --- SGTC-w with cosine / random priority (flagship comparison) ---
preds = []
for H in Hs:
    Hc, c = sgtc_w_compress(H, H.shape[0] // ratio, f, return_counts=True,
                            scores=cosine_scores(H))
    preds.append(float(weighted_pool(Hc, c) @ w))
rows.append(eval_variant("sgtc_w_cos", preds))

rngw = np.random.RandomState(42)
preds = []
for H in Hs:
    Hc, c = sgtc_w_compress(H, H.shape[0] // ratio, f, return_counts=True,
                            scores=rngw.permutation(H.shape[0]).astype(np.float32))
    preds.append(float(weighted_pool(Hc, c) @ w))
rows.append(eval_variant("sgtc_w_rand", preds))

# --- attention priority: needs scores from GPU pass (run_attention_scores) ---
att_dir = config.FEATS / "attention_scores_test"
if any(att_dir.glob("*.npy")):
    preds = []
    for i, H in enumerate(Hs):
        p = att_dir / (ids[i] + ".npy")
        preds.append(f(sgtc_compress(H, H.shape[0] // ratio, f, scores=np.load(p))) if p.exists()
                     else np.nan)
    ok = ~np.isnan(preds)
    m = utterance_metrics(ys[ok], np.array(preds)[ok])
    rows.append({"variant": "attention", **m})
    print(f"attention     LCC={m['lcc']:.4f} SRCC={m['srcc']:.4f} RMSE={m['rmse']:.4f} "
          f"(n={int(ok.sum())})", flush=True)

pd.DataFrame(rows).to_csv(config.RESULTS / "tables" / "table2.csv", index=False)
print("saved table2.csv")
