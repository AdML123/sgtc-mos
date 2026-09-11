"""R4 (E1) corrected: MLP-head control, TRAIN on train features, EVAL on test.

Requires GPU-pre-extracted train caches: w2v2_main_train (exists),
w2v2_main_{merge2,merge3,prune2}_train (re-extracted in the revision GPU queue).

Run:  python -m sgtc.evaluation.run_mlp_head
Output: results/tables/mlp_head.csv  (five seeds {41..45}, mean+-std)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sgtc import config
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.train_regression_head import pool

main = config.DATASETS["main"]
tr_df = pd.read_csv(main["label_train"])
te_df = pd.read_csv(main["label_test"])
tr_map = dict(zip(tr_df["utt_id"], tr_df["mos"]))
te_map = dict(zip(te_df["utt_id"], te_df["mos"]))
SEEDS = [41, 42, 43, 44, 45]


def load_pooled(feats_dir, label_map):
    d = config.FEATS / feats_dir
    X, y, ids = [], [], []
    for p in sorted(d.glob("*.npy")):
        if p.stem in label_map:
            X.append(pool(np.load(p).astype(np.float32)))
            y.append(label_map[p.stem])
            ids.append(p.stem)
    return np.stack(X), np.array(y), ids


def train_mlp_predict(X_tr, y_tr, X_te, seed, hidden=256, max_epoch=300):
    torch.manual_seed(seed)
    n = len(y_tr)
    idx = torch.randperm(n)
    n_val = max(1, n // 10)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X_tr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, patience = 1e9, None, 0
    for _ in range(max_epoch):
        net.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(Xt[tr_i]).squeeze(-1), yt[tr_i])
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(net(Xt[val_i]).squeeze(-1), yt[val_i]).item()
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(X_te, dtype=torch.float32)).squeeze(-1).numpy()


CONFIGS = [("baseline", "w2v2_main_train", "w2v2_main_test"),
           ("merge2", "w2v2_main_merge2_train", "w2v2_main_merge2_test"),
           ("merge3", "w2v2_main_merge3_train", "w2v2_main_merge3_test"),
           ("prune2", "w2v2_main_prune2_train", "w2v2_main_prune2_test"),
           ("merge2", "w2v2_main_merge2_train", "w2v2_main_merge2_test")]

rows = []
done = set()
for name, tr_d, te_d in CONFIGS:
    if name in done:
        continue
    done.add(name)
    X_tr, y_tr, _ = load_pooled(tr_d, tr_map)
    X_te, y_te, _ = load_pooled(te_d, te_map)
    per = [utterance_metrics(y_te, train_mlp_predict(X_tr, y_tr, X_te, s)) for s in SEEDS]
    rows.append({"method": name,
                 "lcc_mean": float(np.mean([m["lcc"] for m in per])),
                 "lcc_std": float(np.std([m["lcc"] for m in per])),
                 "srcc_mean": float(np.mean([m["srcc"] for m in per])),
                 "rmse_mean": float(np.mean([m["rmse"] for m in per]))})
    print(f"MLP {name}: LCC={rows[-1]['lcc_mean']:.4f}±{rows[-1]['lcc_std']:.4f} "
          f"(train n={len(y_tr)}, test n={len(y_te)})", flush=True)

pd.DataFrame(rows).to_csv(config.RESULTS / "tables" / "mlp_head.csv", index=False)
print("MLP HEAD (correct protocol) DONE")
