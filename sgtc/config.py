"""Global paths, model ids and hyperparameters. All experiment scripts import from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # D:\paper51
DATA = ROOT / "data"
FEATS = ROOT / "features"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"                            # local weight copies (gitignored)

DATASETS = {
    # VoiceMOS 2022 main track = BVCC data, Zenodo record 6572573 (public)
    "main": {"wav_train": DATA / "voicemos2022/wavs/train",
             "wav_test": DATA / "voicemos2022/wavs/test",
             "label_train": DATA / "voicemos2022/train_mos.csv",
             "label_test": DATA / "voicemos2022/test_mos.csv"},
    "ood": {"wav": DATA / "voicemos2022_ood/wavs", "label": DATA / "voicemos2022_ood/labels.csv"},
}

# local dirs (hub download is blocked by network; see logs/data_provenance.md)
BACKBONES = {"w2v2": str(MODELS / "w2v2"),
             "hubert": str(MODELS / "hubert"),
             "wavlm": str(MODELS / "wavlm")}

HIDDEN = 768
SEED = 42

# regression head (spec 3.2: stat-pooling + single linear, ridge closed form)
HEAD = {"ridge_lambda": 1e-3}
RATIOS = [2, 3, 4]           # compression ratios
N_SEEDS_SHUFFLE = 5          # shuffle diagnosis seeds
LAYER_SUBSET_N = 200         # utterances for layer-wise analysis
DEVICE = "cuda"              # fallback "cpu" per spec risk table
