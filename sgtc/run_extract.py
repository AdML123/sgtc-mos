"""Stage A bulk extraction runner: w2v2 main-track train/test (+layer-wise subset)."""
import time

from sgtc import config
from sgtc.extract_features import extract_set

t0 = time.time()
main = config.DATASETS["main"]

# 1) full train + test, last layer
extract_set(main["wav_train"], config.FEATS / "w2v2_main_train", "w2v2", device=config.DEVICE)
extract_set(main["wav_test"], config.FEATS / "w2v2_main_test", "w2v2", device=config.DEVICE)

# 2) layer-wise on the first 200 test utterances (13 hidden states each)
import pandas as pd
ids = pd.read_csv(main["label_test"])["utt_id"].tolist()[:200]
keep = set(i[:-4] if i.endswith(".wav") else i for i in ids)
extract_set(main["wav_test"], config.FEATS / "w2v2_main_test_layers", "w2v2",
            device=config.DEVICE, all_layers=True, utt_filter=lambda u: u in keep)

print(f"ALL EXTRACTION DONE in {time.time()-t0:.0f}s")
