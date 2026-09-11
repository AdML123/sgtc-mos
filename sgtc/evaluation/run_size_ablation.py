"""R5 (E2): training-set size ablation, linear ridge head.

Subsamples {25%, 50%, 100%} of train utterances for baseline vs merge2
features, retrains the ridge head, evaluates on the full test set.

Run:  python -m sgtc.evaluation.run_size_ablation
Output: results/tables/size_ablation.csv
"""
import numpy as np
import pandas as pd

from sgtc import config
from sgtc.evaluation.metrics import lcc, utterance_metrics
from sgtc.train_regression_head import load_head, make_predictor, pool

main = config.DATASETS["main"]
tr_df = pd.read_csv(main["label_train"])
te_df = pd.read_csv(main["label_test"])
te_map = dict(zip(te_df["utt_id"], te_df["mos"]))

rng = np.random.RandomState(42)
order = rng.permutation(len(tr_df))

rows = []
for frac, n in [("25%", 566), ("50%", 1132), ("100%", 2254)]:
    sub = tr_df.iloc[sorted(order[:n])]
    for meth, tr_dir in [("baseline", "w2v2_main_train"), ("merge2", "w2v2_main_merge2_train")]:
        cache = {p.stem: pool(np.load(p).astype(np.float32))
                 for p in (config.FEATS / tr_dir).glob("*.npy")}
        X = np.stack([cache[u] for u in sub["utt_id"] if u in cache])
        y = sub[[u in cache for u in sub["utt_id"]]]["mos"].to_numpy(dtype=np.float32)
        w = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
        preds, ys = [], []
        for p in sorted((config.FEATS / "w2v2_main_test").glob("*.npy")):
            if meth == "merge2":
                pm = config.FEATS / "w2v2_main_merge2_test" / p.name
                if not pm.exists():
                    continue
                H = np.load(pm).astype(np.float32)
            else:
                H = np.load(p).astype(np.float32)
            preds.append(float(pool(H) @ w))
            ys.append(te_map[p.stem])
        m = utterance_metrics(ys, preds)
        rows.append({"frac": frac, "n_train": int(len(y)), "method": meth, **m})
        print(f"{frac} {meth}: n={len(y)} LCC={m['lcc']:.4f}", flush=True)

pd.DataFrame(rows).to_csv(config.RESULTS / "tables" / "size_ablation.csv", index=False)
# gain (merge - baseline) per fraction
piv = pd.DataFrame(rows).pivot(index="frac", columns="method", values="lcc")
piv["gain"] = piv["merge2"] - piv["baseline"]
print(piv)
print("SIZE ABLATION DONE")
