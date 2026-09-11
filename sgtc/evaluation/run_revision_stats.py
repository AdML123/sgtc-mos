"""Revision experiments R2 (E4 stats), R3 (NISQA calibration), R5 (size ablation).

Run:  python -m sgtc.evaluation.run_revision_stats
Outputs: results/tables/revision_stats.json, nisqa_calibration.json,
         size_ablation.csv
"""
import json

import numpy as np
import pandas as pd

from sgtc import config
from sgtc.evaluation.metrics import lcc, utterance_metrics
from sgtc.evaluation.stat_test import lcc_gap_bootstrap, paired_err_ttest
from sgtc.train_regression_head import load_head, make_predictor, train_head

main = config.DATASETS["main"]
label_df = pd.read_csv(main["label_test"])
label_map = dict(zip(label_df["utt_id"], label_df["mos"]))


def preds_of(name):
    return np.load(config.RESULTS / "preds" / (name + ".npy"))


ys = np.array([label_map[u] for u in sorted(label_map)])
by_len = len(ys)

# ---------- E4: merge vs prune paired tests; merge vs baseline bootstrap ----------
out = {"merge_vs_prune": {}, "merge_vs_base": {}}
for ratio in [2, 3, 4]:
    rn = {2: "Two", 3: "Three", 4: "Four"}[ratio]
    pm, pp, pb = preds_of(f"merge{ratio}"), preds_of(f"prune{ratio}"), preds_of("baseline")
    out["merge_vs_prune"][ratio] = paired_err_ttest(pm, pp, ys)
    out["merge_vs_base"][ratio] = lcc_gap_bootstrap(pm, pb, ys, n_boot=2000)
    print(f"r={ratio}: merge-vs-prune p={out['merge_vs_prune'][ratio]['p']:.2e}, "
          f"merge-base LCC gap 95%CI={out['merge_vs_base'][ratio]['ci95']}", flush=True)
(config.RESULTS / "tables" / "revision_stats.json").write_text(json.dumps(out, indent=2))

# ---------- E3: NISQA calibration (intercept on 100 held-out utterances) ----------
nisqa_labels = pd.read_csv(r"D:\paper51\data\nisqa\labels.csv")
nisqa_mos = dict(zip(nisqa_labels["utt_id"], nisqa_labels["mos"]))
feat_dirs = {"baseline": "w2v2_nisqa", "merge2": "w2v2_nisqa_merge2",
             "prune2": "w2v2_nisqa_prune2"}
heads = {"baseline": "head_w2v2_main.npz", "merge2": "head_merge2.npz",
         "prune2": "head_prune2.npz"}
sys_rows = []
for meth, fdir in feat_dirs.items():
    f = make_predictor(load_head(config.FEATS / heads[meth]))
    pr = []
    for p in sorted((config.FEATS / fdir).glob("*.npy")):
        pr.append({"utt": p.stem, "pred": f(np.load(p).astype(np.float32)),
                   "mos": nisqa_mos[p.stem]})
    d = pd.DataFrame(pr)
    # calibration set: first 100 of LIVETALK ordering (utterance-stable rule)
    d = d.sort_values("utt").reset_index(drop=True)
    cal, ev = d.iloc[:100], d.iloc[100:]
    for mode in ["none", "intercept"]:
        if mode == "intercept":
            b = float(np.mean(cal["mos"] - cal["pred"]))
            e = ev.assign(pred=ev["pred"] + b)
        else:
            e = ev
        m = utterance_metrics(e["mos"], e["pred"])
        sys_rows.append({"method": meth, "calibration": mode, **m})
        print(f"nisqa {meth} cal={mode}: LCC={m['lcc']:.3f} RMSE={m['rmse']:.3f}", flush=True)
(config.RESULTS / "tables" / "nisqa_calibration.json").write_text(json.dumps(sys_rows, indent=2))

# ---------- E2 lives in run_size_ablation.py (needs merge2 train cache re-extracted on GPU) ----------
print("REVISION STATS DONE")
