"""Fig.3 redesign (RF3): LCC vs compression factor, vertical single-column.

Triple encoding (color x linestyle x marker), direct curve labels instead of a
legend box, five-seed +-1 std band for neuron merging when seed_stability.csv
exists. Single column 3.5 x 2.6 in.
"""
import json
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
fig, ax = plt.subplots(figsize=(3.5, 2.6))

x, y = series("merge")
if std:
    e = [std.get(r, 0.0) for r in x]
    ax.fill_between(x, np.array(y) - e, np.array(y) + e, color=C_MERGE, alpha=0.18, lw=0)
ax.plot(x, y, "-o", color=C_MERGE, ms=4, lw=1.4)
ax.annotate("neuron merging", (x[-1], y[-1]), xytext=(4, 0),
            textcoords="offset points", fontsize=7, color=C_MERGE, va="center")

x, y = series("prune")
ax.plot(x, y, "--s", color=C_PRUNE, ms=3.6, lw=1.3)
ax.annotate("magnitude pruning", (x[-1], y[-1]), xytext=(4, 0),
            textcoords="offset points", fontsize=7, color=C_PRUNE, va="center")

x, y = series("sgtc_w")
ax.plot(x, y, ":^", color=C_TEMP, ms=3.8, lw=1.3)
ax.annotate("temporal (weighted)", (x[-1], y[-1]), xytext=(4, 0),
            textcoords="offset points", fontsize=7, color=C_TEMP, va="center")

ax.axhline(base, color=C_BASE, ls="-.", lw=0.9)
ax.text(2.02, base + 0.004, "uncompressed", fontsize=7, color=C_BASE)

ax.set_xlabel("feed-forward compression factor $r$")
ax.set_ylabel("utterance LCC")
ax.set_xticks([2, 3, 4])
ax.set_xlim(1.9, 3.85)
ax.set_ylim(0.55, 0.86)
fig.tight_layout(pad=0.4)
fig.savefig(figs / "fig3.pdf")
plt.close(fig)
print("fig3 saved (vertical, direct labels, band=%s)" % bool(std))
