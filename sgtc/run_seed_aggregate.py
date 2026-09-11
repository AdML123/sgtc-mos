"""Rebuild seed_stability.csv with full metrics (lcc/srcc/rmse means, lcc std).

Each seed has its own surgery-extracted test cache and ridge head
(seed 42 = the canonical pair, 41/43/44/45 = revision runs).

Run:  python -m sgtc.run_seed_aggregate
"""
import numpy as np
import pandas as pd

from sgtc import config
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.train_regression_head import load_head, make_predictor

main = config.DATASETS["main"]
label_map = dict(zip(pd.read_csv(main["label_test"])["utt_id"],
                     pd.read_csv(main["label_test"])["mos"]))


def eval_head(feats_name, head_name):
    d = config.FEATS / feats_name
    f = make_predictor(load_head(config.FEATS / head_name))
    preds, ys = [], []
    for p in sorted(d.glob("*.npy")):
        preds.append(f(np.load(p).astype(np.float32)))
        ys.append(label_map[p.stem])
    return utterance_metrics(ys, preds)


rows = []
for ratio in [2, 3, 4]:
    per = []
    for seed in [42, 41, 43, 44, 45]:
        name = "merge%d" % ratio if seed == 42 else "merge%d_s%d" % (ratio, seed)
        head = "head_%s.npz" % name
        hp = config.FEATS / head
        if hp.exists():
            per.append(eval_head("w2v2_main_%s_test" % name, head))
    rows.append({"ratio": ratio, "n_seeds": len(per),
                 "lcc_mean": float(np.mean([m["lcc"] for m in per])),
                 "lcc_std": float(np.std([m["lcc"] for m in per])),
                 "srcc_mean": float(np.mean([m["srcc"] for m in per])),
                 "rmse_mean": float(np.mean([m["rmse"] for m in per]))})
    print(f"r={ratio}: LCC={rows[-1]['lcc_mean']:.4f}±{rows[-1]['lcc_std']:.4f} "
          f"SRCC={rows[-1]['srcc_mean']:.4f} RMSE={rows[-1]['rmse_mean']:.4f} "
          f"({len(per)} seeds)", flush=True)

pd.DataFrame(rows).to_csv(config.RESULTS / "tables" / "seed_stability.csv",
                          index=False)
print("seed_stability.csv rebuilt with full metrics")
