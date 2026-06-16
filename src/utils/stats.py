"""Bootstrap CIs and paired significance tests (Phase 9).

Real implementations live here; phase scripts call them. Kept torch-free.
"""
from __future__ import annotations

import numpy as np


def mean_std(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def bootstrap_ci(values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the mean of ``values``."""
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    boots = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean()), float(lo), float(hi)


def paired_test(a, b):
    """Paired comparison (Q-PlasmaNet vs MLP head). Returns Wilcoxon + paired t.

    scipy is imported lazily so audit-only environments don't need it.
    """
    from scipy import stats

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:  # all-zero differences etc.
        w_stat, w_p = float("nan"), float("nan")
    return {
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "mean_diff": float(a.mean() - b.mean()),
    }
