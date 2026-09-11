"""Fig.2 (RF5): representative-utterance redundancy evidence, two panels.

Panel (a): mel spectrogram of the test utterance whose MOS is closest to the
corpus median. Panel (b): cosine similarity between adjacent encoder frames of
the same utterance, the quantitative redundancy behind the axis diagnosis.
Records the utterance and its summary statistics in
results/tables/mel_utterance.json so the manuscript macros stay single-source.

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
    dur = float(len(y) / sr)

    feats = config.FEATS / "w2v2_main_test" / (str(row["utt_id"]) + ".npy")
    H = np.load(feats).astype(np.float32)
    a, b = H[:-1], H[1:]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    t_cos = np.arange(len(cos)) * 0.02     # wav2vec2 frame rate 50 Hz
    mean_cos = float(cos.mean())
    frac09 = float((cos >= 0.9).mean())

    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "svg.fonttype": "none",
                         "font.size": 8, "axes.linewidth": 0.7,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.5, 1.5),
                                   gridspec_kw={"height_ratios": [1.5, 1.0]})
    im = ax1.imshow(S_db, origin="lower", aspect="auto", cmap="magma",
                    extent=[0, dur, 0, 64])
    ax1.set_ylabel("mel bin")
    ax1.tick_params(labelbottom=False)
    cb = fig.colorbar(im, ax=ax1, pad=0.02)
    cb.set_label("dB", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    ax2.plot(t_cos, cos, lw=0.8, color="#0072B2")
    ax2.axhline(0.9, color="0.4", lw=0.6, ls=":")
    ax2.set_ylim(0.5, 1.02)
    ax2.set_ylabel("adjacent cosine")
    ax2.set_xlabel("time (s)")
    fig.tight_layout(pad=0.3, h_pad=0.3)
    fig.savefig(str(Path(r"D:\paper51\paper\figs") / "fig2.svg"),
                bbox_inches="tight")
    fig.savefig(str(Path(r"D:\paper51\paper\figs") / "fig2.pdf"),
                bbox_inches="tight")
    fig.savefig(str(Path(r"D:\paper51\paper\figs") / "fig2.png"),
                dpi=600, bbox_inches="tight")
    plt.close(fig)

    (config.RESULTS / "tables" / "mel_utterance.json").write_text(
        json.dumps({"utt_id": str(row["utt_id"]), "mos": float(row["mos"]),
                    "duration_s": dur, "mean_adj_cosine": mean_cos,
                    "frac_above_09": frac09}))
    print(f"fig2 saved for utt {row['utt_id']} (MOS {row['mos']:.2f}, "
          f"mean cos {mean_cos:.3f}, frac>0.9 {frac09:.3f})")


if __name__ == "__main__":
    main()
