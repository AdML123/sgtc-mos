"""Stage A: encoder forward pass -> cached fp16 features (.npy per utterance).

All downstream diagnosis/compression runs on the caches, never re-running the
encoder (spec section 4).
"""
import numpy as np
import torch
import librosa
from pathlib import Path
from transformers import Wav2Vec2Model, HubertModel, WavLMModel

MODEL_CLS = {"w2v2": Wav2Vec2Model, "hubert": HubertModel, "wavlm": WavLMModel}


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=16000)
    return y


def load_model(backbone: str, device: str):
    from sgtc import config
    model = MODEL_CLS[backbone].from_pretrained(config.BACKBONES[backbone])
    if device == "cuda":
        model = model.half()
    return model.to(device).eval()


@torch.no_grad()
def encode_one(model, wav: np.ndarray, device: str, all_layers: bool = False) -> list:
    x = torch.tensor(wav).unsqueeze(0).to(device)
    if device == "cuda":
        x = x.half()
    out = model(x, output_hidden_states=all_layers)
    hs = out.hidden_states if all_layers else (out.last_hidden_state,)
    return [h.squeeze(0).float().cpu().numpy().astype(np.float16) for h in hs]


def extract_set(wav_dir: Path, out_dir: Path, backbone: str, device="cuda",
                all_layers=False, limit=None, utt_filter=None, model=None):
    """utt_filter: optional callable(utt_id)->bool to select a subset by id.
    `model`: optionally pass a pre-loaded (possibly surgically modified) model."""
    owns_model = model is None
    if owns_model:
        model = load_model(backbone, device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(Path(wav_dir).glob("*.wav"))
    if utt_filter is not None:
        wavs = [w for w in wavs if utt_filter(w.stem)]
    if limit:
        wavs = wavs[:limit]
    for i, w in enumerate(wavs):
        if (out_dir / (w.stem + ".npy")).exists():
            continue
        layers = encode_one(model, load_audio(w), device, all_layers=all_layers)
        if all_layers:
            for li, h in enumerate(layers):
                (out_dir / f"layer{li}").mkdir(parents=True, exist_ok=True)
                np.save(out_dir / f"layer{li}" / (w.stem + ".npy"), h)
        else:
            np.save(out_dir / (w.stem + ".npy"), layers[0])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)}", flush=True)
    if owns_model:
        del model
        torch.cuda.empty_cache()
        print(f"extract_set done: {len(wavs)} utts -> {out_dir}", flush=True)
    else:
        print(f"extract_set done (external model): {len(wavs)} utts -> {out_dir}", flush=True)
