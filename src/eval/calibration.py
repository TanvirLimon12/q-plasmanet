"""Phase 3 — calibration: ECE, Brier score, reliability bins."""
from __future__ import annotations

import numpy as np


def compute_calibration(y_true, y_prob, n_bins: int = 15) -> dict:
    """Expected Calibration Error, Brier score, and reliability-curve data."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return {"ece": float("nan"), "brier": float("nan"), "bins": []}

    brier = float(np.mean((y_prob - y_true) ** 2))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    n = y_true.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        w = float(mask.sum()) / n
        ece += w * abs(acc - conf)
        bins.append({"lo": float(lo), "hi": float(hi), "conf": conf, "acc": acc, "weight": w})

    return {"ece": float(ece), "brier": brier, "bins": bins}
