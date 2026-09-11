"""SSL-MOS original three-branch head control (pooled-statistics equivalent).

The three-branch head of the VoiceMOS 2022 baseline is linear in the pooled
statistics: the per-frame branch and the mean branch together form one weight
on the pooled mean, and the std branch forms the weight on the pooled std
(see the module docstring derivation). We therefore train the identical
function on precomputed pooled statistics with the published Adam recipe,
which is mathematically equivalent to the frame-level loop but seconds fast.

Run:  python -m sgtc.evaluation.run_sslmos_head
Output: results/tables/sslmos_head.csv
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sgtc import config
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.train_regression_head import pool

main = config.DATASETS["main"]
tr_map = dict(zip(pd.read_csv(main["label_train"])["utt_id"],
                  pd.read_csv(main["label_train"])["mos"]))
te_map = dict(zip(pd.read_csv(main["label_test"])["utt_id"],
                  pd.read_csv(main["label_test"])["mos"]))
SEEDS = [41, 42, 43, 44, 45]


class SslMosHeadPooled(nn.Module):
    """Three-branch linear head on z = (mean; std).

    Frame branch + mean branch both act on the pooled mean, so their weights
    add; the std branch acts on the pooled std. This is the exact function
    class of the frame-level SSL-MOS head, trained with its Adam recipe."""

    def __init__(self, hidden_dim=768):
        super().__init__()
        self.weighted_layer = nn.Linear(hidden_dim, 1)   # frame branch
        self.mean_layer = nn.Linear(hidden_dim, 1)       # mean branch
        self.std_layer = nn.Linear(hidden_dim, 1)        # std branch

    def forward(self, z):                    # z: (B, 2N) = (mean; std)
        mu, sd = z.chunk(2, dim=1)
        return (self.weighted_layer(mu) + self.mean_layer(mu)
                + self.std_layer(sd)).squeeze(-1)


def load_pooled(feats_name, label_map):
    """z = (mean; std with ddof=1), matching torch.std defaults of the
    frame-level SSL-MOS head."""
    d = config.FEATS / feats_name
    Z, ys = [], []
    for p in sorted(d.glob("*.npy")):
        if p.stem in label_map:
            H = np.load(p).astype(np.float32)
            Z.append(np.concatenate([H.mean(axis=0), H.std(axis=0, ddof=1)]))
            ys.append(label_map[p.stem])
    return np.stack(Z), np.array(ys, dtype=np.float32)


def train_predict(Z_tr, y_tr, Z_te, seed, max_epoch=500, lr=1e-3):
    torch.manual_seed(seed)
    n = len(y_tr)
    idx = torch.randperm(n)
    n_val = max(1, n // 10)
    val_i, tr_i = idx[:n_val], idx[n_val:]
    X = torch.tensor(Z_tr)
    Xv = X[val_i]
    yv = torch.tensor(y_tr[val_i])
    net = SslMosHeadPooled()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, patience = 1e9, None, 0
    for _ in range(max_epoch):
        net.train()
        opt.zero_grad()
        out = net(X[tr_i])
        loss = nn.functional.mse_loss(out, torch.tensor(y_tr[tr_i]))
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(net(Xv), yv).item()
        if vl < best - 1e-6:
            best, patience = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Z_te)).tolist()


CONFIGS = [("baseline", "w2v2_main_train", "w2v2_main_test"),
           ("merge2", "w2v2_main_merge2_train", "w2v2_main_merge2_test"),
           ("merge3", "w2v2_main_merge3_train", "w2v2_main_merge3_test"),
           ("prune2", "w2v2_main_prune2_train", "w2v2_main_prune2_test")]


def run():
    rows = []
    for name, tr_d, te_d in CONFIGS:
        Z_tr, y_tr = load_pooled(tr_d, tr_map)
        Z_te, y_te = load_pooled(te_d, te_map)
        per = []
        for seed in SEEDS:
            preds = train_predict(Z_tr, y_tr, Z_te, seed)
            per.append(utterance_metrics(y_te, preds))
        rows.append({"method": name,
                     "lcc_mean": float(np.mean([m["lcc"] for m in per])),
                     "lcc_std": float(np.std([m["lcc"] for m in per])),
                     "srcc_mean": float(np.mean([m["srcc"] for m in per])),
                     "rmse_mean": float(np.mean([m["rmse"] for m in per]))})
        print(f"SSL-MOS head {name}: LCC={rows[-1]['lcc_mean']:.4f}"
              f"±{rows[-1]['lcc_std']:.4f}", flush=True)
    # unregularized closed form on the same pooled class (isolates the
    # regularization, not the architecture, as the load-bearing component)
    for name, tr_d, te_d in [("baseline_ols", "w2v2_main_train", "w2v2_main_test"),
                             ("merge2_ols", "w2v2_main_merge2_train",
                              "w2v2_main_merge2_test")]:
        Z_tr, y_tr = load_pooled(tr_d, tr_map)
        Z_te, y_te = load_pooled(te_d, te_map)
        w = np.linalg.solve(Z_tr.T @ Z_tr, Z_tr.T @ y_tr)
        m = utterance_metrics(y_te, Z_te @ w)
        rows.append({"method": name, "lcc_mean": float(m["lcc"]),
                     "lcc_std": 0.0, "srcc_mean": float(m["srcc"]),
                     "rmse_mean": float(m["rmse"])})
        print(f"OLS closed form {name}: LCC={m['lcc']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(config.RESULTS / "tables" / "sslmos_head.csv",
                              index=False)
    print("SSL-MOS HEAD CONTROL DONE")


if __name__ == "__main__":
    run()
