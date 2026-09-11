"""Utterance-level MOS prediction metrics: LCC, SRCC, RMSE."""
import numpy as np
from scipy.stats import pearsonr, spearmanr


def lcc(y_true, y_pred) -> float:
    return float(pearsonr(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))[0])


def srcc(y_true, y_pred) -> float:
    return float(spearmanr(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))[0])


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def utterance_metrics(y_true, y_pred) -> dict:
    return {"lcc": lcc(y_true, y_pred), "srcc": srcc(y_true, y_pred), "rmse": rmse(y_true, y_pred)}
