"""Task 4.3 runner: prune/merge surgery -> re-extract -> retrain head -> cleanup.

For each (mode, ratio): fresh model -> surgery(K=3072/ratio) -> extract train
(temp) -> train head -> extract test (kept) -> delete train cache.
"""
import shutil
import time
from pathlib import Path

import torch

from sgtc import config
from sgtc.compression.channel_pruning import surgically_compress_ffn
from sgtc.extract_features import load_model, extract_set
from sgtc.train_regression_head import train_head

main = config.DATASETS["main"]


def no_op_check():
    """K=3072 surgery must reproduce the original forward."""
    from sgtc.extract_features import load_audio, encode_one
    m0 = load_model("w2v2", "cpu").float().eval()
    w = next(main["wav_test"].glob("*.wav"))
    x = load_audio(w)
    y0 = encode_one(m0, x, "cpu")[0]
    surgically_compress_ffn(m0, 3072, "prune")
    y1 = encode_one(m0, x, "cpu")[0]
    assert y0.shape == y1.shape
    err = float(np.abs(y0.astype(np.float32) - y1.astype(np.float32)).max())
    print(f"no-op check max|diff| = {err:.2e}")
    assert err < 5e-3, "K=3072 surgery changed the function unexpectedly"


import numpy as np
no_op_check()

for mode in ["prune", "merge"]:
    for ratio in [2, 3, 4]:
        K = 3072 // ratio
        tag = f"{mode}{ratio}"
        te_dir = config.FEATS / f"w2v2_main_{tag}_test"
        head_npz = config.FEATS / f"head_{tag}.npz"
        if head_npz.exists() and any(te_dir.glob("*.npy")):
            print(f"[{tag}] already done, skip", flush=True)
            continue
        t0 = time.time()
        model = load_model("w2v2", config.DEVICE)
        surgically_compress_ffn(model, K, mode)
        tr_dir = config.FEATS / f"w2v2_main_{tag}_train"
        te_dir = config.FEATS / f"w2v2_main_{tag}_test"
        extract_set(main["wav_train"], tr_dir, "w2v2", device=config.DEVICE, model=model)
        train_head(tr_dir, main["label_train"], config.FEATS / f"head_{tag}.npz")
        shutil.rmtree(tr_dir)                       # disk control
        extract_set(main["wav_test"], te_dir, "w2v2", device=config.DEVICE, model=model)
        del model
        torch.cuda.empty_cache()
        print(f"[{tag}] K={K} done in {time.time()-t0:.0f}s", flush=True)
print("SURGERY RUNNER DONE")
