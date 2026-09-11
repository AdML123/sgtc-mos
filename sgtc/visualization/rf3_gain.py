"""Fig.3 (RF3 final): LCC vs compression factor, legend row ABOVE the body.

Four series with color x linestyle x marker triple encoding; five-seed +-1 std
band for neuron merging. Legend sits in its own top row, panels carry no
overlapping legend. Single column 3.5 x 2.9 in.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sgtc import config

C_MERGE, C_PRUNE, C_TEMP, C_BASE = "#009E73", "#D55E00", "#0072B2", "0.35"
figs = Path(r"D:\paper51\paper\figs")
t1 = pd.read_csv(config.RESULTS / "tables" / "table1.csv")


def series(method):
    r = t1[t1.method == method].sort_values("ratio")
    return list(r["ratio"]), list(r["lcc"])


base = float(t1[t1.method == "baseline"]["lcc"].iloc[0])
std = {}
ss = config.RESULTS / "tables" / "seed_stability.csv"
if ss.exists():
    for _, row in pd.read_csv(ss).iterrows():
        std[int(row["ratio"])] = float(row["lcc_std"])

mpl.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(3.5, 2.1))
gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 3.0], hspace=0.25,
                      left=0.15, right=0.97, top=0.98, bottom=0.13)
axL = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

# ---- legend row (above the body) ----
axL.axis("off")
handles = [
    mpl.lines.Line2D([], [], color=C_MERGE, ls="-", marker="o", ms=4, lw=1.4),
    mpl.lines.Line2D([], [], color=C_PRUNE, ls="--", marker="s", ms=3.6, lw=1.3),
    mpl.lines.Line2D([], [], color=C_TEMP, ls=":", marker="^", ms=3.8, lw=1.3),
    mpl.lines.Line2D([], [], color=C_BASE, ls="-.", lw=0.9),
]
axL.legend(handles, ["neuron merging", "magnitude pruning",
                     "temporal (weighted)", "uncompressed"],
           loc="center", ncol=2, fontsize=8, frameon=False,
           handlelength=2.4, columnspacing=1.1, borderaxespad=0)

# ---- body ----
x, y = series("merge")
if std:
    e = [std.get(r, 0.0) for r in x]
    ax.fill_between(x, np.array(y) - e, np.array(y) + e, color=C_MERGE,
                    alpha=0.18, lw=0)
ax.plot(x, y, "-o", color=C_MERGE, ms=4, lw=1.4)
x, y = series("prune")
ax.plot(x, y, "--s", color=C_PRUNE, ms=3.6, lw=1.3)
x, y = series("sgtc_w")
ax.plot(x, y, ":^", color=C_TEMP, ms=3.8, lw=1.3)
ax.axhline(base, color=C_BASE, ls="-.", lw=0.9)

ax.set_xlabel("feed-forward compression factor $r$")
ax.set_ylabel("utterance LCC")
ax.set_xticks([2, 3, 4])
ax.set_xlim(1.9, 3.85)
ax.set_ylim(0.55, 0.86)
fig.savefig(figs / "fig3.pdf")
plt.close(fig)
print("fig3 saved (legend above body)")
