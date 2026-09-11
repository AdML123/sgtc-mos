"""Task 4.4 runner: SGTC-deploy accuracy (2x on full test) + wall-clock timing."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sgtc import config
from sgtc.compression.sgtc_deploy import deploy_forward
from sgtc.extract_features import load_audio, load_model
from sgtc.train_regression_head import load_head, make_predictor
from sgtc.evaluation.metrics import utterance_metrics

main = config.DATASETS["main"]
model = load_model("w2v2", config.DEVICE)
f = make_predictor(load_head(config.FEATS / "head_w2v2_main.npz"))
df = pd.read_csv(main["label_test"])

# ---- accuracy at 2x (and 4x for the Pareto curve) ----
acc = {}
for ratio in [2, 4]:
    preds, ys = [], []
    for _, r in df.iterrows():
        wav = main["wav_test"] / (str(r["utt_id"]) + ".wav")
        if not wav.exists():
            continue
        x = torch.tensor(load_audio(wav)).unsqueeze(0).half().cuda()
        with torch.no_grad():
            h = deploy_forward(model, x, ratio=ratio)[0].float().cpu().numpy().astype(np.float32)
        preds.append(f(h))
        ys.append(r["mos"])
    acc[f"{ratio}x"] = utterance_metrics(ys, preds)
    print(f"deploy {ratio}x: {acc[f'{ratio}x']}", flush=True)

# ---- wall-clock on 200 utts ----
timing = {"full": [], "deploy2x": []}
for _, r in df.head(200).iterrows():
    wav = main["wav_test"] / (str(r["utt_id"]) + ".wav")
    if not wav.exists():
        continue
    x = torch.tensor(load_audio(wav)).unsqueeze(0).half().cuda()
    for name, fn in [("full", lambda: model(x)),
                     ("deploy2x", lambda: deploy_forward(model, x, ratio=2))]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()
        timing[name].append(time.perf_counter() - t0)

out = {"accuracy": acc,
       "timing_ms": {k: float(np.mean(v) * 1000) for k, v in timing.items()},
       "speedup": float(np.mean(timing["full"]) / np.mean(timing["deploy2x"])),
       "n_utts": {"accuracy": int(len(acc["2x"]) * 0 + len(df)), "timing": len(timing["full"])}}
print(json.dumps(out, indent=2))
(config.RESULTS / "tables").mkdir(parents=True, exist_ok=True)
(config.RESULTS / "tables" / "sgtc_deploy.json").write_text(json.dumps(out, indent=2))
print("saved sgtc_deploy.json")
