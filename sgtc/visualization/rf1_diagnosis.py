"""Fig.1 redesign (RF1): axis diagnosis, vertical two-panel composite.

(a) per-utterance prediction-shift distributions, time vs channel shuffle
(b) LCC drop by shuffle granularity and dimension
Single column 3.5in wide, panels stacked (vertical layout), IEEE style:
Okabe-Ito colors + hatch/line dual encoding (grayscale-safe), 8pt text.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sgtc import config

C_TIME, C_CHAN = "#0072B2", "#D55E00"          # Okabe-Ito blue / vermillion
figs = Path(r"D:\paper51\paper\figs")

# shifts were cached by the v1 figure script
shifts = np.load(config.RESULTS / "preds" / "fig1a_shifts.npy")
st, sc = shifts[0], shifts[1]
diag = json.loads((config.RESULTS / "tables" / "shuffle_diagnosis.json").read_text())
base = diag["original"]["lcc_mean"]

mpl.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.8))

# ---- (a) shift distributions ----
bins = np.linspace(-4, 4, 65)
ax1.hist(np.clip(st, -4, 4), bins=bins, histtype="step", lw=1.4, color=C_TIME,
         label=f"time (|shift| {np.abs(st).mean():.2f})")
ax1.hist(np.clip(sc, -4, 4), bins=bins, color=C_CHAN, alpha=0.55,
         hatch="////", edgecolor="none",
         label=f"channel (|shift| {np.abs(sc).mean():.1f})")
ax1.set_ylabel("utterances")
ax1.set_yticks([])
ax1.set_xlabel("signed prediction shift after shuffling (MOS)")
ax1.legend(fontsize=7, loc="upper left")
ax1.text(0.02, 0.86, "(a)", transform=ax1.transAxes, fontweight="bold")

# ---- (b) LCC drop by granularity ----
labels = ["global", "block-50", "adjacent"]
time_d = [base - diag["time_shuffle"]["lcc_mean"],
          base - diag["block_shuffle"]["lcc_mean"], 0.0]
chan_d = [base - diag["chan_shuffle"]["lcc_mean"], np.nan, np.nan]
x = np.arange(len(labels))
ax2.bar(x - 0.17, time_d, width=0.32, facecolor="none", edgecolor=C_TIME,
        lw=1.2, label="time shuffle")
ax2.bar(x[:1] + 0.17, chan_d[0], width=0.32, color=C_CHAN, hatch="////",
        edgecolor="none", label="channel shuffle")
ax2.set_xticks(x, labels)
ax2.set_ylabel("LCC drop vs. original")
ax2.legend(fontsize=7, loc="upper right")
ax2.text(0.02, 0.86, "(b)", transform=ax2.transAxes, fontweight="bold")

fig.tight_layout(pad=0.4, h_pad=1.2)
fig.savefig(figs / "fig1.pdf")
plt.close(fig)
print("fig1 saved (vertical composite)")
