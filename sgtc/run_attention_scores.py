"""Extract per-frame attention-received scores on the test set (ablation var 4).

score_t = column-sum of the final encoder layer's attention (averaged over
heads): how much the rest of the sequence attends to frame t. Low-received
frames are treated as low sensitivity by the ablation variant.
"""
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Model

from sgtc import config
from sgtc.extract_features import load_audio, load_model

main = config.DATASETS["main"]
out_dir = config.FEATS / "attention_scores_test"
out_dir.mkdir(parents=True, exist_ok=True)

model = Wav2Vec2Model.from_pretrained(config.BACKBONES["w2v2"], attn_implementation="eager")
if config.DEVICE == "cuda":
    model = model.half()
model = model.to(config.DEVICE).eval()
df = pd.read_csv(main["label_test"])
done = 0
for _, r in df.iterrows():
    dest = out_dir / (str(r["utt_id"]) + ".npy")
    if dest.exists():
        continue
    wav = main["wav_test"] / (str(r["utt_id"]) + ".wav")
    if not wav.exists():
        continue
    x = torch.tensor(load_audio(wav)).unsqueeze(0).half().cuda()
    with torch.no_grad():
        out = model(x, output_attentions=True)
    att = out.attentions[-1][0]          # (heads, T, T)
    score = att.sum(dim=(0, 1)).float().cpu().numpy().astype(np.float32)  # received attention
    np.save(dest, score)
    done += 1
    if done % 200 == 0:
        print(done, flush=True)
print("attention scores done:", done)
