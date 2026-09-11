"""Fig.4: layer-wise shuffle diagnosis (d_time vs d_chan across 13 layers)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sgtc import config

C_TIME, C_CHAN = "#0072B2", "#D55E00"
d = json.loads((config.RESULTS / "tables" / "layer_diagnosis.json").read_text())
layers = sorted(int(k) for k in d)
dt = [d[str(l)]["d_time"] for l in layers]
dc = [d[str(l)]["d_chan"] for l in layers]

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.plot(layers, dc, "o-", color=C_CHAN, ms=3, lw=1.2, label="channel-shuffle drop")
ax.plot(layers, dt, "s-", color=C_TIME, ms=3, lw=1.2, label="time-shuffle drop")
ax.set_xlabel("wav2vec2 layer")
ax.set_ylabel("LCC drop vs. original")
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
ax.legend(frameon=False, fontsize=7)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(Path(r"D:\paper51\paper\figs") / "fig4.pdf")
plt.close(fig)
print("fig4 saved")
