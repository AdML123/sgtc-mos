"""GPU tail of the revision queue (run after seed stability):

1. Re-extract train caches for merge2/merge3/prune2 (needed by the corrected
   MLP control and the size ablation; they were deleted for disk control).
2. Measure wall-clock of the merged (r=2) encoder vs the baseline encoder and
   append merge2_speedup to results/tables/sgtc_deploy.json.

Run:  python -m sgtc.run_gpu_tail
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from sgtc import config
from sgtc.compression.channel_pruning import surgically_compress_ffn
from sgtc.extract_features import extract_set, load_audio, load_model

main = config.DATASETS["main"]

# ---------- 1. train caches ----------
for mode, ratio in [("merge", 2), ("merge", 3), ("prune", 2)]:
    tag = f"{mode}{ratio}"
    tr_dir = config.FEATS / f"w2v2_main_{tag}_train"
    if any(tr_dir.glob("*.npy")):
        print(f"[{tag}] train cache exists, skip", flush=True)
        continue
    model = load_model("w2v2", config.DEVICE)
    surgically_compress_ffn(model, 3072 // ratio, mode)
    extract_set(main["wav_train"], tr_dir, "w2v2", device=config.DEVICE, model=model)
    del model
    torch.cuda.empty_cache()
    print(f"[{tag}] train cache restored", flush=True)

# ---------- 2. wall-clock: baseline vs merged encoder ----------
df = pd.read_csv(main["label_test"])
base_m = load_model("w2v2", config.DEVICE)
merge_m = load_model("w2v2", config.DEVICE)
surgically_compress_ffn(merge_m, 1536, "merge")
timing = {"base": [], "merge2": []}
for _, r in df.head(200).iterrows():
    wav = main["wav_test"] / (str(r["utt_id"]) + ".wav")
    if not wav.exists():
        continue
    x = torch.tensor(load_audio(wav)).unsqueeze(0).half().cuda()
    for name, mdl in [("base", base_m), ("merge2", merge_m)]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            mdl(x)
        torch.cuda.synchronize()
        timing[name].append(time.perf_counter() - t0)

out_path = config.RESULTS / "tables" / "sgtc_deploy.json"
j = json.loads(out_path.read_text())
j["merge2_timing_ms"] = {"base": float(np.mean(timing["base"]) * 1000),
                         "merge2": float(np.mean(timing["merge2"]) * 1000)}
j["merge2_speedup"] = float(np.mean(timing["base"]) / np.mean(timing["merge2"]))
out_path.write_text(json.dumps(j, indent=2))
print(f"merge2 encoder wall-clock speedup: {j['merge2_speedup']:.2f} "
      f"({j['merge2_timing_ms']['merge2']:.1f} ms vs {j['merge2_timing_ms']['base']:.1f} ms)")
print("GPU TAIL DONE")
