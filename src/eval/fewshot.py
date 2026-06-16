"""Supplementary analysis — few-shot learning on generic (ImageNet) ResNet-18
features: does the quantum head help when only k labels/class are available?

Mirrors the data-scaling pipeline but samples a fixed k per class instead of a
fraction. Uses frozen ImageNet ResNet-18 PCA(d24) features (the harder, generic
regime where configurations differentiate). Trains logistic / matched-MLP /
q-reupload on k labels/class, evaluates the full PCMMD test split, R draws per k.

Output: results/fewshot.csv (k_per_class, model, f1_mean, f1_std, auc_mean, auc_std)

  python -m src.eval.fewshot --config configs/backbones.yaml configs/quantum.yaml

Note: recreated to close the reproducibility gap flagged in docs/CODE_AUDIT.md
(results/fewshot.csv previously had no generator in the source tree).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import compute_metrics  # noqa: E402
from src.features.precompute import load_features  # noqa: E402
from src.models.classical_heads import mlp_for_param_budget  # noqa: E402
from src.models.quantum_head import build_reupload_head  # noqa: E402
from src.train.engine import predict, train_loop  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402


def _fit_eval(model, Xk, yk, Xte, yte, epochs=60):
    dl = DataLoader(TensorDataset(torch.tensor(Xk, dtype=torch.float32),
                                  torch.tensor(yk, dtype=torch.long)),
                    batch_size=min(64, len(Xk)), shuffle=True)
    best, _ = train_loop(model, dl, None, epochs=epochs, lr=0.03, seed=0,
                         device=torch.device("cpu"), verbose=False)
    model.load_state_dict(best)
    te = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                                  torch.tensor(yte, dtype=torch.long)), batch_size=256)
    yt, pt = predict(model, te, torch.device("cpu"))
    return compute_metrics(yt, pt)


def run_fewshot(cfg, k_values=(5, 10, 20, 40), draws=5, seed=0, backbone="resnet18"):
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results"); results.mkdir(parents=True, exist_ok=True)
    Xtr, ytr = load_features(cfg, backbone, 24, seed, "train")
    Xte, yte = load_features(cfg, backbone, 24, seed, "test")
    if Xtr is None or Xte is None:
        raise FileNotFoundError(
            f"need {backbone} d24 features; run precompute --backbones {backbone} --dims 24.")
    ytr = ytr.astype(int); yte = yte.astype(int)
    i0, i1 = np.where(ytr == 0)[0], np.where(ytr == 1)[0]

    rows = []
    for k in k_values:
        scores = {m: {"f1": [], "auc": []} for m in ("logistic", "mlp_matched", "q_reupload")}
        for d in range(draws):
            rng = np.random.default_rng(d)
            sel = np.concatenate([rng.choice(i0, min(k, len(i0)), replace=False),
                                  rng.choice(i1, min(k, len(i1)), replace=False)])
            Xk, yk = Xtr[sel], ytr[sel]
            lr = LogisticRegression(max_iter=2000).fit(Xk, yk)
            mlr = compute_metrics(yte, lr.predict_proba(Xte)[:, 1])
            scores["logistic"]["f1"].append(mlr["f1"]); scores["logistic"]["auc"].append(mlr["auc"])
            mm = _fit_eval(mlp_for_param_budget(24, 178, 2), Xk, yk, Xte, yte)
            scores["mlp_matched"]["f1"].append(mm["f1"]); scores["mlp_matched"]["auc"].append(mm["auc"])
            q, _ = build_reupload_head(8, 3, "circular", "lightning.qubit", "adjoint", verify=False)
            qm = _fit_eval(q, Xk, yk, Xte, yte)
            scores["q_reupload"]["f1"].append(qm["f1"]); scores["q_reupload"]["auc"].append(qm["auc"])
        for m, v in scores.items():
            rows.append({"k_per_class": k, "model": m,
                         "f1_mean": float(np.mean(v["f1"])), "f1_std": float(np.std(v["f1"])),
                         "auc_mean": float(np.mean(v["auc"])), "auc_std": float(np.std(v["auc"]))})
        print(f"[fewshot] k={k}: " +
              " ".join(f"{m}={np.mean(v['f1']):.3f}" for m, v in scores.items()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(results / "fewshot.csv", index=False)
    print(f"[fewshot] wrote results/fewshot.csv")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Few-shot on generic ResNet-18 features (supplementary).")
    add_common_args(p)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--draws", type=int, default=5)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml", "configs/quantum.yaml"])
    run_fewshot(cfg, draws=args.draws, backbone=args.backbone)


if __name__ == "__main__":
    main()
