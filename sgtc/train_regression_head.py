"""Ridge linear head on statistical pooling (spec 3.2).

f(H) = concat(mean_t h_t, std_t h_t) @ w ; closed-form ridge (equivalent to
least squares on the linear head, deterministic, trains in seconds on cached
features).
"""
from pathlib import Path

import numpy as np
import pandas as pd


def pool(H: np.ndarray) -> np.ndarray:
    """H: (T, N) float32 -> (2N,) mean||std concat."""
    return np.concatenate([H.mean(axis=0), H.std(axis=0)])


def train_head(feats_dir, label_csv, out_npz, ridge_lambda: float = 1e-3) -> np.ndarray:
    feats_dir = Path(feats_dir)
    df = label_csv if isinstance(label_csv, pd.DataFrame) else pd.read_csv(label_csv)
    X, y, missing = [], [], 0
    for _, r in df.iterrows():
        f = feats_dir / (str(r["utt_id"]) + ".npy")
        if f.exists():
            X.append(pool(np.load(f).astype(np.float32)))
            y.append(float(r["mos"]))
        else:
            missing += 1
    X, y = np.stack(X), np.asarray(y, dtype=np.float32)
    w = np.linalg.solve(X.T @ X + ridge_lambda * np.eye(X.shape[1]), X.T @ y)
    if out_npz:
        np.savez(out_npz, w=w)
        print(f"head trained on {len(y)} utts ({missing} missing), dim={X.shape[1]}")
    return w


def load_head(npz_path) -> np.ndarray:
    return np.load(npz_path)["w"]


def make_predictor(w: np.ndarray):
    def f(H: np.ndarray) -> float:
        return float(pool(H.astype(np.float32)) @ w)
    return f
