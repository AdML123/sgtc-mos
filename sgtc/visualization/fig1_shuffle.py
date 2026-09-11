"""Fig.1 data prep + plotting: shuffle diagnosis.

(a) per-utterance prediction shift distributions under time vs channel shuffle
(b) LCC drop by shuffle granularity and dimension.
Inputs: features/w2v2_main_test + head; results/tables/shuffle_diagnosis.json.
Outputs: paper/figs/fig1a.pdf, fig1b.pdf (vector, IEEE column width).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sgtc import config
from sgtc.train_regression_head import load_head, make_predictor
from sgtc.diagnosis.shuffle_test import block_shuffle

C_TIME, C_CHAN = "#0072B2", "#D55E00"          # Okabe-Ito
main = config.DATASETS["main"]
figs = Path(r"D:\paper51\paper\figs")
figs.mkdir(parents=True, exist_ok=True)

w = load_head(config.FEATS / "head_w2v2_main.npz")
f = make_predictor(w)
label_map = dict(zip(pd.read_csv(main["label_test"])["utt_id"],
                     pd.read_csv(main["label_test"])["mos"]))

# ---------- (a) per-utterance shifts ----------
rng = np.random.RandomState(42)
shifts_t, shifts_c = [], []
for p in sorted((config.FEATS / "w2v2_main_test").glob("*.npy")):
    H = np.load(p).astype(np.float32)
    y0 = f(H)
    Ht = H[rng.permutation(H.shape[0])]
    Hc = H[:, rng.permutation(H.shape[1])]
    shifts_t.append(f(Ht) - y0)
    shifts_c.append(f(Hc) - y0)
shifts_t, shifts_c = np.array(shifts_t), np.array(shifts_c)
np.save(config.RESULTS / "preds" / "fig1a_shifts.npy",
        np.vstack([shifts_t, shifts_c]))

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 1.9))
bins = np.linspace(-4, 4, 65)
ax.hist(np.clip(shifts_t, -4, 4), bins=bins, color=C_TIME, alpha=0.85,
        label=f"time (|shift| mean {np.abs(shifts_t).mean():.2f})")
ax.hist(np.clip(shifts_c, -4, 4), bins=bins, color=C_CHAN, alpha=0.55,
        label=f"channel (|shift| mean {np.abs(shifts_c).mean():.2f})")
ax.set_xlabel("prediction shift after shuffling (MOS)")
ax.set_ylabel("utterances")
ax.legend(frameon=False, fontsize=7)
ax.set_yticks([])
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(figs / "fig1a.pdf")
plt.close(fig)

# ---------- (b) LCC drop by granularity ----------
diag = json.loads((config.RESULTS / "tables" / "shuffle_diagnosis.json").read_text())
base = diag["original"]["lcc_mean"]
gran = [("global", "time"), ("global", "chan"),
        ("block(50)", "time"), ("block(50)", "chan"),
        ("adjacent", "time"), ("adjacent", "chan")]
vals = [base - diag["time_shuffle"]["lcc_mean"], base - diag["chan_shuffle"]["lcc_mean"],
        base - diag["block_shuffle"]["lcc_mean"], np.nan,
        0.0, 0.0]   # adjacent swap on this head is an exact no-op; chan granularity degenerates
labels = ["global", "block-50", "adjacent"]
time_d = [base - diag["time_shuffle"]["lcc_mean"],
          base - diag["block_shuffle"]["lcc_mean"], 0.0]
chan_d = [base - diag["chan_shuffle"]["lcc_mean"], np.nan, np.nan]

fig, ax = plt.subplots(figsize=(3.5, 1.9))
x = np.arange(len(labels))
ax.bar(x - 0.18, time_d, width=0.32, color=C_TIME, label="time shuffle")
ax.bar(x[:1] + 0.18, chan_d[0], width=0.32, color=C_CHAN, label="channel shuffle")
ax.set_xticks(x, labels)
ax.set_ylabel("LCC drop vs. original")
ax.legend(frameon=False, fontsize=7)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(figs / "fig1b.pdf")
plt.close(fig)
print("fig1a/fig1b saved")
