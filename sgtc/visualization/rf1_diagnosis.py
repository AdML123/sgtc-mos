"""Fig.1 (RF1 final): axis diagnosis, vertical layout, legend ABOVE the body.

Legend occupies its own top row of the figure; the two panels below carry no
legend, so the legend never overlaps panel content. Single column 3.5in.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sgtc import config

C_TIME, C_CHAN = "#0072B2", "#D55E00"
figs = Path(r"D:\paper51\paper\figs")

shifts = np.load(config.RESULTS / "preds" / "fig1a_shifts.npy")
st, sc = shifts[0], shifts[1]
diag = json.loads((config.RESULTS / "tables" / "shuffle_diagnosis.json").read_text())
base = diag["original"]["lcc_mean"]

mpl.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(3.5, 2.5))
gs = fig.add_gridspec(3, 1, height_ratios=[0.6, 2.6, 1.9], hspace=0.45,
                      left=0.16, right=0.96, top=0.97, bottom=0.08)
axL = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])

# ---- legend row (above both panels) ----
h_time = mpl.lines.Line2D([], [], color=C_TIME, lw=1.4)
h_chan = mpl.patches.Patch(color=C_CHAN, hatch="////", alpha=0.55)
axL.axis("off")
axL.legend([h_time, h_chan], ["time shuffle", "channel shuffle"],
           loc="center", ncol=2, fontsize=8, frameon=False,
           handlelength=2.2, columnspacing=1.2, borderaxespad=0)

# ---- (a) shift distributions ----
bins = np.linspace(-4, 4, 65)
ax1.hist(np.clip(st, -4, 4), bins=bins, histtype="step", lw=1.4, color=C_TIME)
ax1.hist(np.clip(sc, -4, 4), bins=bins, color=C_CHAN, alpha=0.55,
         hatch="////", edgecolor="none")
ax1.set_ylabel("utterances")
ax1.set_yticks([])
ax1.set_xlabel("signed prediction shift after shuffling (MOS)")
ax1.text(0.02, 0.88, "(a) time shift $|\\Delta|$=0.00, channel $|\\Delta|$=%.1f"
         % np.abs(sc).mean(), transform=ax1.transAxes, fontsize=7)

# ---- (b) LCC drop by granularity ----
labels = ["global", "block-50", "adjacent"]
time_d = [base - diag["time_shuffle"]["lcc_mean"],
          base - diag["block_shuffle"]["lcc_mean"], 0.0]
chan_d = [base - diag["chan_shuffle"]["lcc_mean"], np.nan, np.nan]
x = np.arange(len(labels))
ax2.bar(x - 0.17, time_d, width=0.32, facecolor="none", edgecolor=C_TIME, lw=1.2)
ax2.bar(x[:1] + 0.17, chan_d[0], width=0.32, color=C_CHAN, hatch="////",
        edgecolor="none")
ax2.set_xticks(x, labels)
ax2.set_ylabel("LCC drop vs. original")
ax2.text(0.02, 0.88, "(b)", transform=ax2.transAxes, fontweight="bold")

fig.savefig(figs / "fig1.pdf")
plt.close(fig)
print("fig1 saved (legend above body)")
