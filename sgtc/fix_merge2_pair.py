"""Fix merge2 pair inconsistency: re-extract merge2 test with the current
(MiniBatch k-means) implementation to match the restored train cache, retrain
the head, and refresh Table I row + preds + the merge-vs-sgtc test.

Run:  python -m sgtc.fix_merge2_pair
"""
import json

import numpy as np
import pandas as pd
import torch

from sgtc import config
from sgtc.compression.channel_pruning import surgically_compress_ffn
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.evaluation.stat_test import paired_err_ttest
from sgtc.extract_features import extract_set, load_model
from sgtc.train_regression_head import train_head

main = config.DATASETS["main"]
te_dir = config.FEATS / "w2v2_main_merge2_test"
tr_dir = config.FEATS / "w2v2_main_merge2_train"
head_npz = config.FEATS / "head_merge2.npz"

model = load_model("w2v2", config.DEVICE)
surgically_compress_ffn(model, 1536, "merge", seed=42)
for d in [te_dir, tr_dir]:
    for f in d.glob("*.npy"):
        f.unlink()
extract_set(main["wav_train"], tr_dir, "w2v2", device=config.DEVICE, model=model)
train_head(tr_dir, main["label_train"], head_npz)
extract_set(main["wav_test"], te_dir, "w2v2", device=config.DEVICE, model=model)
del model
torch.cuda.empty_cache()

# evaluate and refresh table1/preds
from sgtc.train_regression_head import load_head, make_predictor
f = make_predictor(load_head(head_npz))
te_map = dict(zip(pd.read_csv(main["label_test"])["utt_id"],
                  pd.read_csv(main["label_test"])["mos"]))
preds, ys = [], []
for p in sorted(te_dir.glob("*.npy")):
    preds.append(f(np.load(p).astype(np.float32)))
    ys.append(te_map[p.stem])
m = utterance_metrics(ys, preds)
np.save(config.RESULTS / "preds" / "merge2.npy", np.asarray(preds, dtype=np.float32))
print("consistent merge2:", m, flush=True)

t1p = config.RESULTS / "tables" / "table1.csv"
t1 = pd.read_csv(t1p)
mask = (t1.method == "merge") & (t1.ratio == 2)
t1.loc[mask, ["lcc", "srcc", "rmse"]] = [m["lcc"], m["srcc"], m["rmse"]]
# refresh merge_vs_sgtc t-test at r=2
sg = np.load(config.RESULTS / "preds" / "sgtc2.npy")
t = paired_err_ttest(np.asarray(preds), sg, ys)
tm = t1.method == "merge_vs_sgtc_t"
t1.loc[tm & (t1.ratio == 2), ["t", "p"]] = [t["t"], t["p"]]
t1.to_csv(t1p, index=False)

# refresh seed-stability seed-42 entries (per_seed lives only in printouts; csv stores aggregates)
sp = config.RESULTS / "tables" / "seed_stability.csv"
ss = pd.read_csv(sp)
ss.loc[ss.ratio == 2, "lcc_mean"] = None  # recompute below
rows = []
for ratio in [2, 3, 4]:
    per = []
    for seed in [41, 42, 43, 44, 45]:
        name = "merge%d" % ratio if seed == 42 else "merge%d_s%d" % (ratio, seed)
        hp = config.FEATS / ("head_%s.npz" % name)
        d = config.FEATS / ("w2v2_main_%s_test" % name)
        if not hp.exists():
            continue
        fs = make_predictor(load_head(hp))
        pr, yy = [], []
        for p in sorted(d.glob("*.npy")):
            pr.append(fs(np.load(p).astype(np.float32)))
            yy.append(te_map[p.stem])
        per.append(utterance_metrics(yy, pr)["lcc"])
    rows.append({"ratio": ratio, "n_seeds": len(per),
                 "lcc_mean": float(np.mean(per)), "lcc_std": float(np.std(per))})
    print(f"5-seed r={ratio}: {rows[-1]['lcc_mean']:.4f}±{rows[-1]['lcc_std']:.4f}", flush=True)
pd.DataFrame(rows).to_csv(sp, index=False)
print("MERGE2 PAIR FIXED")
