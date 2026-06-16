"""Phase 3 — classification metrics: acc / F1 / AUC / PR-AUC / balanced acc.

Binary task (plasma=1 vs non_plasma=0). Inputs are y_true (0/1) and y_prob
(probability of class 1). AUC/PR-AUC degrade gracefully to NaN on a single class.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return {k: float("nan") for k in ("acc", "f1", "auc", "pr_auc", "balanced_acc")}

    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) > 1

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
