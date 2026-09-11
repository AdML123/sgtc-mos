"""Fig.4 redesign (RF4): layer-wise diagnosis, vertical single-column 3.5x2.4in.

Channel drop as filled-marker line with light fill; time drop as open squares
on the exact-invariance zero line. IEEE style, 8pt.
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
fig, ax = plt.subplots(figsize=(3.5, 2.4))

ax.fill_between(layers, 0, dc, color=C_CHAN, alpha=0.15, lw=0)
ax.plot(layers, dc, "-o", color=C_CHAN, ms=3.4, lw=1.3,
        label="channel shuffle")
ax.plot(layers, dt, "s", mfc="none", mec=C_TIME, ms=3.4, ls="none",
        label="time shuffle")
ax.axhline(0, color=C_TIME, lw=0.8, ls=":")
ax.text(6.1, 0.018, "exact invariance", fontsize=7, color=C_TIME)

ax.set_xlabel("wav2vec2 layer")
ax.set_ylabel("LCC drop vs. original")
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
ax.set_ylim(-0.04, 0.82)
ax.legend(fontsize=7, loc="upper center", ncol=2)
fig.tight_layout(pad=0.4)
fig.savefig(figs / "fig4.pdf")
plt.close(fig)
print("fig4 saved (vertical)")
