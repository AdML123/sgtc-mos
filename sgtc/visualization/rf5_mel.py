"""Fig.2 (RF5): mel spectrogram of a representative real test utterance.

Picks the test utterance whose MOS is closest to the corpus median, renders
its mel spectrogram (librosa, dB scale) as a single-column vertical figure,
and records the chosen utterance in results/tables/mel_utterance.json so that
the manuscript macro \MelMos stays single-source.

Run:  python -m sgtc.visualization.rf5_mel
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from sgtc import config


def main():
    main = config.DATASETS["main"]
    df = pd.read_csv(main["label_test"])
    median = float(df["mos"].median())
    row = df.iloc[int((df["mos"] - median).abs().argsort().iloc[0])]
    wav = main["wav_test"] / (str(row["utt_id"]) + ".wav")

    import librosa
    y, sr = librosa.load(str(wav), sr=16000)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160,
                                       n_mels=64, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "svg.fonttype": "none",
                         "font.size": 8, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    im = ax.imshow(S_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_xticks(np.linspace(0, S.shape[1] - 1, 5))
    ax.set_xticklabels([f"{t:.1f}" for t in np.linspace(0, len(y) / sr, 5)])
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("dB", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(Path(r"D:\paper51\paper\figs") / "fig2.svg"),
                bbox_inches="tight")
    fig.savefig(str(Path(r"D:\paper51\paper\figs") / "fig2.pdf"),
                bbox_inches="tight")
    fig.savefig(str(Path(r"D:\paper51\paper\figs") / "fig2.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)

    (config.RESULTS / "tables" / "mel_utterance.json").write_text(
        json.dumps({"utt_id": str(row["utt_id"]), "mos": float(row["mos"]),
                    "duration_s": float(len(y) / sr)}))
    print(f"fig2 saved for utt {row['utt_id']} (MOS {row['mos']:.2f})")


if __name__ == "__main__":
    main()
