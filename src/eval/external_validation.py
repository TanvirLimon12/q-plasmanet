"""Phase 8 — external validation on MedMNIST (BloodMNIST, PathMNIST).

Validation, not a contribution (MedMNIST is heavily benchmarked) — used only to
show the head comparison generalizes beyond PCMMD. Both datasets are CC BY 4.0.
Multi-class is reframed to a balanced binary sub-task (two classes).

Pipeline mirrors the main one: frozen ImageNet backbone -> PCA(d24) [-pi,pi] ->
{logistic (linear-head analog), param-matched MLP, re-uploading VQC}. Reports
test F1/AUC/acc per dataset per head -> results/external_validation.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import compute_metrics  # noqa: E402
from src.models.backbones import build_backbone, freeze  # noqa: E402
from src.models.classical_heads import mlp_for_param_budget  # noqa: E402
from src.models.quantum_head import build_reupload_head  # noqa: E402
from src.train.engine import get_device, predict, train_loop  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402

_PI = float(np.pi)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DATASETS = {"bloodmnist": "BloodMNIST", "pathmnist": "PathMNIST"}


def _load_split(cls, split, classes, size=64):
    ds = cls(split=split, download=True, size=size)
    X = np.asarray(ds.imgs)                       # (N,H,W,3) uint8
    y = np.asarray(ds.labels).reshape(-1)
    mask = np.isin(y, classes)
    X, y = X[mask], y[mask]
    y = (y == classes[1]).astype(int)             # binary: classes[1]->1 else 0
    return X, y


@torch.no_grad()
def _features(backbone, X, device, bs=256):
    backbone.eval()
    t = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    out = []
    for i in range(0, len(t), bs):
        out.append(backbone(t[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(out)


def _train_eval_torch(model, Xtr, ytr, Xte, yte, device, epochs=40, lr=0.02, seed=0):
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.long)), batch_size=64, shuffle=True)
    best, _ = train_loop(model, dl, None, epochs=epochs, lr=lr, seed=seed, device=device, verbose=False)
    model.load_state_dict(best)
    te = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                                  torch.tensor(yte, dtype=torch.long)), batch_size=256)
    yt, pt = predict(model, te, device)
    return compute_metrics(yt, pt)


def run_external(cfg, datasets=None, classes=(0, 1), backbone_name="resnet18",
                 dim=24, smoke=False, limit=None, seeds=(0, 1, 2)) -> pd.DataFrame:
    import medmnist

    repo_root = Path(cfg._repo_root)
    results_dir = repo_root / cfg.get_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()                    # backbone feature extraction (GPU/MPS ok)
    head_dev = torch.device("cpu")           # heads incl. quantum train on CPU (lightning)
    datasets = datasets or list(DATASETS)
    if smoke:
        limit = limit or 200

    backbone, _ = build_backbone(backbone_name, cfg, pretrained=True)
    freeze(backbone); backbone.to(device)

    rows = []
    for name in datasets:
        cls = getattr(medmnist, DATASETS[name])
        Xtr, ytr = _load_split(cls, "train", classes)
        Xte, yte = _load_split(cls, "test", classes)
        if limit:
            Xtr, ytr = Xtr[:limit], ytr[:limit]
            Xte, yte = Xte[: max(40, limit // 2)], yte[: max(40, limit // 2)]

        Ftr, Fte = _features(backbone, Xtr, device), _features(backbone, Xte, device)
        tf = Pipeline([("pca", PCA(n_components=min(dim, Ftr.shape[1]), random_state=0)),
                       ("scale", MinMaxScaler((-_PI, _PI)))]).fit(Ftr)
        Ztr, Zte = tf.transform(Ftr), tf.transform(Fte)

        # logistic (linear-head analog) — deterministic, once
        lr = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
        m = compute_metrics(yte, lr.predict_proba(Zte)[:, 1])
        rows.append({"dataset": name, "head": "logistic", "seed": 0, "n_train": len(ytr), **m})

        ep = 1 if smoke else 40
        sd = [0] if smoke else list(seeds)               # torch heads over seeds for CIs
        for s in sd:
            m = _train_eval_torch(mlp_for_param_budget(dim, 178, 2), Ztr, ytr, Zte, yte, head_dev, epochs=ep, seed=s)
            rows.append({"dataset": name, "head": "mlp_param_matched", "seed": s, "n_train": len(ytr), **m})
            q, _ = build_reupload_head(8, 3, "circular", "lightning.qubit", "adjoint", verify=False)
            m = _train_eval_torch(q, Ztr, ytr, Zte, yte, head_dev, epochs=ep, seed=s)
            rows.append({"dataset": name, "head": "q_plasmanet", "seed": s, "n_train": len(ytr), **m})
        print(f"[external] {name}: done ({len(sd)} seeds)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "external_validation.csv", index=False)
    print(f"[external] wrote {results_dir/'external_validation.csv'}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="External validation on MedMNIST (Phase 8).")
    add_common_args(p)
    p.add_argument("--datasets", nargs="*", default=None, choices=list(DATASETS))
    p.add_argument("--classes", nargs=2, type=int, default=[0, 1])
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml", "configs/quantum.yaml"])
    run_external(cfg, datasets=args.datasets, classes=tuple(args.classes), smoke=args.smoke)


if __name__ == "__main__":
    main()
