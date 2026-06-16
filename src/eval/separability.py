"""Q1 analysis — data separability, to substantiate "near-separable ⇒ no quantum
advantage" (instead of asserting it).

Reports, per feature set: linear-probe accuracy/F1 (logistic regression test
performance — high ⇒ classes near-linearly separable), PCA participation-ratio
intrinsic dimension, and the variance captured in 2 PCs. Compares the PCMMD-trained
cnn d24 features (used by the main quantum head) against generic ImageNet resnet18
d24, showing the trained features are highly separable so capacity is not limiting.

Output: results/separability.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import compute_metrics  # noqa: E402
from src.features.precompute import load_features  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402


def participation_ratio(X) -> float:
    """Intrinsic dimension proxy: (Σλ)² / Σλ² of the feature covariance spectrum."""
    lam = PCA().fit(X).explained_variance_
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))


def run_separability(cfg, backbones=("cnn", "resnet18"), dim=24, seed=0) -> pd.DataFrame:
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results"); results.mkdir(parents=True, exist_ok=True)
    rows = []
    for bb in backbones:
        Xtr, ytr = load_features(cfg, bb, dim, seed, "train")
        Xte, yte = load_features(cfg, bb, dim, seed, "test")
        if Xtr is None:
            print(f"[separability] no d{dim} features for {bb} — skipped")
            continue
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        m = compute_metrics(yte, clf.predict_proba(Xte)[:, 1])
        pr = participation_ratio(Xtr)
        var2 = float(PCA(n_components=2).fit(Xtr).explained_variance_ratio_.sum())
        rows.append({"backbone": bb, "dim": dim,
                     "linear_probe_f1": round(m["f1"], 4),
                     "linear_probe_acc": round(m["acc"], 4),
                     "linear_probe_auc": round(m["auc"], 4),
                     "participation_ratio": round(pr, 3),
                     "var_in_2pc": round(var2, 4)})
        print(f"[separability] {bb} d{dim}: linear-probe F1={m['f1']:.3f} AUC={m['auc']:.3f} "
              f"intrinsic_dim≈{pr:.2f} var@2PC={var2:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(results / "separability.csv", index=False)
    print(f"[separability] wrote {results/'separability.csv'}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Data separability / intrinsic dimension (Q1).")
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml", "configs/quantum.yaml"])
    run_separability(cfg)


if __name__ == "__main__":
    main()
