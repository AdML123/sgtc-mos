"""Fig.2b: LCC vs feed-forward compression factor (merge/prune/temporal-w)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sgtc import config

C_MERGE, C_PRUNE, C_TEMP = "#009E73", "#D55E00", "#0072B2"
t1 = pd.read_csv(config.RESULTS / "tables" / "table1.csv")


def series(method):
    r = t1[t1.method == method].sort_values("ratio")
    return list(r["ratio"]), list(r["lcc"])


base = float(t1[(t1.method == "baseline")]["lcc"].iloc[0])
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 2.1))
for method, color, label, ls in [
        ("merge", C_MERGE, "neuron merging", "-"),
        ("prune", C_PRUNE, "magnitude pruning", "--"),
        ("sgtc_w", C_TEMP, "temporal (weighted)", ":")]:
    x, y = series(method)
    ax.plot(x, y, ls, marker="o", ms=3.5, lw=1.3, color=color, label=label)
ax.axhline(base, color="gray", lw=0.8, ls="-.")
ax.text(3.55, base - 0.008, "uncompressed", fontsize=7, color="gray", va="top")
ax.set_xlabel("feed-forward compression factor $r$")
ax.set_ylabel("utterance LCC")
ax.set_xticks([2, 3, 4])
ax.set_ylim(0.58, 0.85)
ax.legend(frameon=False, fontsize=7, loc="lower left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(Path(r"D:\paper51\paper\figs") / "fig2b.pdf")
print("fig2b saved")
