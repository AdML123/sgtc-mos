"""Task R6: five-seed k-means stability for neuron merging.

Runs the merge surgery + re-extraction + head retraining for extra seeds
(seed 42 equals the published results), aggregates LCC mean+-std, and refreshes
the random-dropping baseline rows of Table I to five seeds.

Usage:  python -m sgtc.run_seed_stability [seeds...]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sgtc import config
from sgtc.compression.channel_pruning import surgically_compress_ffn
from sgtc.compression.random_drop import random_drop_compress
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.extract_features import extract_set, load_model
from sgtc.train_regression_head import load_head, make_predictor, train_head

SEEDS = [int(s) for s in sys.argv[1:]] or [41, 43, 44, 45]
RATIOS = [2, 3, 4]
main = config.DATASETS["main"]

# ---------- part 1: surgery for each extra seed ----------
for seed in SEEDS:
    if seed == 42:
        continue
    for ratio in RATIOS:
        tag = f"merge{ratio}_s{seed}"
        te_dir = config.FEATS / f"w2v2_main_{tag}_test"
        head_npz = config.FEATS / f"head_{tag}.npz"
        if head_npz.exists() and any(te_dir.glob("*.npy")):
            print(f"[{tag}] exists, skip", flush=True)
            continue
        t0 = time.time()
        model = load_model("w2v2", config.DEVICE)
        surgically_compress_ffn(model, 3072 // ratio, "merge", seed=seed)
        tr_dir = config.FEATS / f"w2v2_main_{tag}_train"
        extract_set(main["wav_train"], tr_dir, "w2v2", device=config.DEVICE, model=model)
        train_head(tr_dir, main["label_train"], head_npz)
        shutil.rmtree(tr_dir)
        extract_set(main["wav_test"], te_dir, "w2v2", device=config.DEVICE, model=model)
        del model
        torch.cuda.empty_cache()
        print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)

# ---------- part 2: aggregate five-seed LCC ----------
label_map = dict(zip(pd.read_csv(main["label_test"])["utt_id"],
                     pd.read_csv(main["label_test"])["mos"]))


def eval_head(feats_name, head_name):
    d = config.FEATS / feats_name
    w = load_head(config.FEATS / head_name)
    f = make_predictor(w)
    preds, ys = [], []
    for p in sorted(d.glob("*.npy")):
        preds.append(f(np.load(p).astype(np.float32)))
        ys.append(label_map[p.stem])
    return utterance_metrics(ys, preds)


rows = []
for ratio in RATIOS:
    per_seed = []
    for seed in [42] + [s for s in SEEDS if s != 42]:
        if seed == 42:
            name, head = f"merge{ratio}", f"head_merge{ratio}.npz"
        else:
            name, head = f"merge{ratio}_s{seed}", f"head_merge{ratio}_s{seed}.npz"
        hp = config.FEATS / head
        if hp.exists():
            m = eval_head(f"w2v2_main_{name}_test", head)
            per_seed.append({"seed": seed, **m})
            print(f"merge r={ratio} seed={seed}: LCC={m['lcc']:.4f}", flush=True)
    lccs = [r["lcc"] for r in per_seed]
    rows.append({"ratio": ratio, "n_seeds": len(per_seed),
                 "lcc_mean": float(np.mean(lccs)), "lcc_std": float(np.std(lccs)),
                 "srcc_mean": float(np.mean([r["srcc"] for r in per_seed])),
                 "rmse_mean": float(np.mean([r["rmse"] for r in per_seed])),
                 "per_seed": per_seed})
out = config.RESULTS / "tables" / "seed_stability.csv"
pd.DataFrame([{k: v for k, v in r.items() if k != "per_seed"} for r in rows]).to_csv(out, index=False)

# ---------- part 3: random dropping refreshed to five seeds ----------
w0 = load_head(config.FEATS / "head_w2v2_main.npz")
f0 = make_predictor(w0)
Hs, ys = [], []
for p in sorted((config.FEATS / "w2v2_main_test").glob("*.npy")):
    Hs.append(np.load(p).astype(np.float32))
    ys.append(label_map[p.stem])
rand_rows = []
for ratio in RATIOS:
    ms = []
    for seed in [41, 42, 43, 44, 45]:
        preds = [f0(random_drop_compress(H, H.shape[0] // ratio, seed=seed)) for H in Hs]
        ms.append(utterance_metrics(ys, preds))
    rand_rows.append({"ratio": ratio,
                      "lcc_mean": float(np.mean([m["lcc"] for m in ms])),
                      "lcc_std": float(np.std([m["lcc"] for m in ms])),
                      "rmse_mean": float(np.mean([m["rmse"] for m in ms]))})
    print(f"random r={ratio}: LCC={rand_rows[-1]['lcc_mean']:.4f}±{rand_rows[-1]['lcc_std']:.4f}",
          flush=True)
(config.RESULTS / "tables" / "random_drop_5seed.json").write_text(
    json.dumps(rand_rows, indent=2))
print("SEED STABILITY DONE")
