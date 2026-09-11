"""Task 3.3v2: layer-wise shuffle diagnosis (paper Fig.4 data).

Per layer l in 0..12: ridge head trained on train-layer-l features, then
time/chan shuffle diagnosis on the 200-utt test subset. Read-only over the
project's own feature caches; writes one summary JSON into results/.
"""
import json

import numpy as np
import pandas as pd

from sgtc import config
from sgtc.diagnosis.shuffle_test import shuffle_diagnosis
from sgtc.train_regression_head import make_predictor, train_head

main = config.DATASETS["main"]
test_df = pd.read_csv(main["label_test"])
mos = dict(zip(test_df["utt_id"], test_df["mos"]))
train_df = pd.read_csv(main["label_train"])

out = {}
for layer in range(13):
    ltr = config.FEATS / "w2v2_main_train_layers" / ("layer" + str(layer))
    lte = config.FEATS / "w2v2_main_test_layers" / ("layer" + str(layer))
    head_npz = config.FEATS / ("head_layer" + str(layer) + ".npz")
    if not head_npz.exists():
        cached = {f.stem for f in ltr.glob("*.npy")}
        train_head(ltr, train_df[train_df["utt_id"].isin(cached)], head_npz)
    f = make_predictor(np.load(head_npz)["w"])
    Hs, ys = [], []
    for p in sorted(lte.glob("*.npy")):
        Hs.append(np.load(p).astype(np.float32))
        ys.append(mos[p.stem])
    res = shuffle_diagnosis(Hs, np.array(ys), f, n_seeds=3)
    orig = float(np.mean([d["lcc"] for d in res["original"]]))
    dt = orig - float(np.mean([d["lcc"] for d in res["time_shuffle"]]))
    dc = orig - float(np.mean([d["lcc"] for d in res["chan_shuffle"]]))
    out[layer] = {"lcc_orig": orig, "d_time": dt, "d_chan": dc}
    print("layer%2d: LCC=%.4f  d_time=%.4f  d_chan=%.4f" % (layer, orig, dt, dc), flush=True)

(config.RESULTS / "tables").mkdir(parents=True, exist_ok=True)
(config.RESULTS / "tables" / "layer_diagnosis.json").write_text(json.dumps(out, indent=2))
print("saved layer_diagnosis.json")
