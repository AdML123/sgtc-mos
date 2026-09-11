"""Fig.4 (RF4 final): layer-wise diagnosis, legend row ABOVE the body.

Channel drop as filled-marker line with light area fill; time drop as open
squares on the exact-invariance zero line. Legend occupies its own top row.
Single column 3.5 x 2.7 in.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sgtc import config

C_CHAN, C_TIME = "#D55E00", "#0072B2"
figs = Path(r"D:\paper51\paper\figs")
d = json.loads((config.RESULTS / "tables" / "layer_diagnosis.json").read_text())
layers = sorted(int(k) for k in d)
dc = np.array([d[str(l)]["d_chan"] for l in layers])
dt = np.array([d[str(l)]["d_time"] for l in layers])

mpl.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(3.5, 2.0))
gs = fig.add_gridspec(2, 1, height_ratios=[0.55, 3.0], hspace=0.25,
                      left=0.15, right=0.97, top=0.98, bottom=0.14)
axL = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

axL.axis("off")
axL.legend(
    [mpl.lines.Line2D([], [], color=C_CHAN, ls="-", marker="o", ms=3.4, lw=1.3),
     mpl.lines.Line2D([], [], color=C_TIME, ls="none", marker="s", ms=3.4,
                      mfc="none", mec=C_TIME)],
    ["channel shuffle", "time shuffle"],
    loc="center", ncol=2, fontsize=8, frameon=False,
    handlelength=2.2, columnspacing=1.2, borderaxespad=0)

ax.fill_between(layers, 0, dc, color=C_CHAN, alpha=0.15, lw=0)
ax.plot(layers, dc, "-o", color=C_CHAN, ms=3.4, lw=1.3)
ax.plot(layers, dt, "s", mfc="none", mec=C_TIME, ms=3.4, ls="none")
ax.axhline(0, color=C_TIME, lw=0.8, ls=":")
ax.text(6.1, 0.018, "exact invariance", fontsize=7, color=C_TIME)

ax.set_xlabel("wav2vec2 layer")
ax.set_ylabel("LCC drop vs. original")
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
ax.set_ylim(-0.04, 0.82)
fig.savefig(figs / "fig4.pdf")
plt.close(fig)
print("fig4 saved (legend above body)")
