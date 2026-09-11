"""Phase 6: NISQA cross-domain evaluation (Table III) + cross-backbone.

Cross-domain protocol: w2v2 head trained on BVCC main track is applied to NISQA
test features (extracted here). Methods at 2x: SGTC / SGTC-w / prune / ToMe.
Cross-backbone (HuBERT/WavLM): extract train/test features, per-backbone ridge
head, shuffle diagnosis + SGTC at 2x/4x.
"""
import json

import numpy as np
import pandas as pd
import torch

from sgtc import config
from sgtc.compression.sgtc import sgtc_compress
from sgtc.compression.sgtc_w import sgtc_w_compress, weighted_pool
from sgtc.compression.tome import tome_compress
from sgtc.extract_features import extract_set, load_audio, load_model
from sgtc.evaluation.metrics import utterance_metrics
from sgtc.train_regression_head import load_head, make_predictor, train_head

RATIO = 2


def nisqa_eval():
    labels = pd.read_csv(r"D:\paper51\data\nisqa\labels.csv")
    want = set(labels["utt_id"])
    w2v_feats = config.FEATS / "w2v2_nisqa"
    prune_feats = config.FEATS / "w2v2_nisqa_prune2"
    merge_feats = config.FEATS / "w2v2_nisqa_merge2"
    if not any(w2v_feats.glob("*.npy")):
        for db in ["NISQA_TEST_FOR", "NISQA_TEST_LIVETALK", "NISQA_TEST_P501"]:
            extract_set(r"D:\paper51\data\nisqa" + "\\" + db + "\\deg", w2v_feats,
                        "w2v2", device=config.DEVICE, utt_filter=lambda u: u in want)
    for feats_dir, mode, K in [(prune_feats, "prune", 1536), (merge_feats, "merge", 1536)]:
        if any(feats_dir.glob("*.npy")):
            continue
        from sgtc.compression.channel_pruning import surgically_compress_ffn
        model = load_model("w2v2", config.DEVICE)
        surgically_compress_ffn(model, K, mode)
        for db in ["NISQA_TEST_FOR", "NISQA_TEST_LIVETALK", "NISQA_TEST_P501"]:
            extract_set(r"D:\paper51\data\nisqa" + "\\" + db + "\\deg", feats_dir,
                        "w2v2", device=config.DEVICE, utt_filter=lambda u: u in want,
                        model=model)
        del model
        torch.cuda.empty_cache()

    w = load_head(config.FEATS / "head_w2v2_main.npz")     # BVCC-trained head
    f = make_predictor(w)
    hp = load_head(config.FEATS / "head_prune2.npz")
    fp = make_predictor(hp)
    hm = load_head(config.FEATS / "head_merge2.npz")
    fmm = make_predictor(hm)
    rows = []
    Hs, ys, Hs_p, Hs_m = [], [], [], []
    for _, r in labels.iterrows():
        p = w2v_feats / (str(r["utt_id"]) + ".npy")
        pp = prune_feats / (str(r["utt_id"]) + ".npy")
        pm = merge_feats / (str(r["utt_id"]) + ".npy")
        if p.exists():
            Hs.append(np.load(p).astype(np.float32))
            ys.append(r["mos"])
            Hs_p.append(np.load(pp).astype(np.float32) if pp.exists() else None)
            Hs_m.append(np.load(pm).astype(np.float32) if pm.exists() else None)
    ys = np.array(ys)
    rows.append({"method": "baseline", **utterance_metrics(ys, [f(H) for H in Hs])})
    rows.append({"method": "sgtc", **utterance_metrics(
        ys, [f(sgtc_compress(H, H.shape[0] // RATIO, f)) for H in Hs])})
    pw = []
    for H in Hs:
        Hc, c = sgtc_w_compress(H, H.shape[0] // RATIO, f, return_counts=True)
        pw.append(float(weighted_pool(Hc, c) @ w))
    rows.append({"method": "sgtc_w", **utterance_metrics(ys, pw)})
    rows.append({"method": "tome", **utterance_metrics(
        ys, [f(tome_compress(H, H.shape[0] // RATIO)) for H in Hs])})
    preds_p, ys_p = [], []
    for H, y in zip(Hs_p, ys):
        if H is not None:
            preds_p.append(fp(H))
            ys_p.append(y)
    rows.append({"method": "prune2", **utterance_metrics(ys_p, preds_p)})
    preds_m, ys_m = [], []
    for H, y in zip(Hs_m, ys):
        if H is not None:
            preds_m.append(fmm(H))
            ys_m.append(y)
    rows.append({"method": "merge2", **utterance_metrics(ys_m, preds_m)})
    pd.DataFrame(rows).to_csv(config.RESULTS / "tables" / "table3.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def backbone_surgery(bb: str, ratio: int = 2):
    """Per-backbone neuron-merging evaluation (generalization of the headline)."""
    import torch
    from sgtc.compression.channel_pruning import surgically_compress_ffn
    main = config.DATASETS["main"]
    feats = config.FEATS / f"{bb}_main_merge{ratio}"
    if not any(feats.glob("*.npy")):
        from sgtc.extract_features import load_model
        model = load_model(bb, config.DEVICE)
        surgically_compress_ffn(model, 3072 // ratio, "merge")
        extract_set(main["wav_train"], feats / "train", bb,
                    device=config.DEVICE, model=model)
        extract_set(main["wav_test"], feats / "test", bb,
                    device=config.DEVICE, model=model)
        del model
        torch.cuda.empty_cache()
    head = config.FEATS / f"head_{bb}_merge{ratio}.npz"
    if not head.exists():
        label_df = pd.read_csv(main["label_train"])
        cached = {p.stem for p in (feats / "train").glob("*.npy")}
        train_head(feats / "train", label_df[label_df["utt_id"].isin(cached)], head)
    import shutil
    shutil.rmtree(feats / "train")          # disk control: keep test only
    wb = load_head(head)
    fb = make_predictor(wb)
    label_map = dict(zip(pd.read_csv(main["label_test"])["utt_id"],
                         pd.read_csv(main["label_test"])["mos"]))
    Hs, ys = [], []
    for p in sorted((feats / "test").glob("*.npy")):
        Hs.append(np.load(p).astype(np.float32))
        ys.append(label_map[p.stem])
    return utterance_metrics(np.array(ys), [fb(H) for H in Hs])


def cross_backbone():
    main = config.DATASETS["main"]
    out = {}
    for bb in ["hubert", "wavlm"]:
        tr = config.FEATS / f"{bb}_main_train"
        te = config.FEATS / f"{bb}_main_test"
        if not any(te.glob("*.npy")):
            extract_set(main["wav_train"], tr, bb, device=config.DEVICE)
            extract_set(main["wav_test"], te, bb, device=config.DEVICE)
        head = config.FEATS / f"head_{bb}_main.npz"
        if not head.exists():
            train_head(tr, main["label_train"], head)
        w = load_head(head)
        f = make_predictor(w)
        label_map = dict(zip(pd.read_csv(main["label_test"])["utt_id"],
                             pd.read_csv(main["label_test"])["mos"]))
        Hs, ys = [], []
        for p in sorted(te.glob("*.npy")):
            Hs.append(np.load(p).astype(np.float32))
            ys.append(label_map[p.stem])
        ys = np.array(ys)
        base_m = utterance_metrics(ys, [f(H) for H in Hs])
        # lightweight diagnosis (3 seeds) for delta_time / delta_chan
        from sgtc.diagnosis.shuffle_test import shuffle_diagnosis
        res = shuffle_diagnosis(Hs, ys, f, n_seeds=3)
        d_time = base_m["lcc"] - np.mean([d["lcc"] for d in res["time_shuffle"]])
        d_chan = base_m["lcc"] - np.mean([d["lcc"] for d in res["chan_shuffle"]])
        entry = {"baseline": base_m, "d_time": float(d_time), "d_chan": float(d_chan)}
        for ratio in [2, 4]:
            preds = [f(sgtc_compress(H, H.shape[0] // ratio, f)) for H in Hs]
            entry[f"sgtc{ratio}"] = utterance_metrics(ys, preds)
        entry["merge2"] = backbone_surgery(bb, 2)
        out[bb] = entry
        print(bb, json.dumps(entry), flush=True)
    (config.RESULTS / "tables" / "cross_backbone.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["nisqa"]:
        nisqa_eval()
    elif sys.argv[1:] == ["backbone"]:
        cross_backbone()
    else:
        nisqa_eval()
        cross_backbone()
