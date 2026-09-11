"""Random frame dropping baseline (uniform, seedable, keeps temporal order)."""
import numpy as np


def random_drop_compress(H: np.ndarray, T_target: int, seed: int = 42) -> np.ndarray:
    T = H.shape[0]
    if T <= T_target:
        return H.astype(np.float32)
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(T, size=T_target, replace=False))
    return H[keep].astype(np.float32)
