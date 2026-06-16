"""Phase 4 — train the quantum head on precomputed features.

Default path: frozen backbone + cached D-dim feature vectors (D = n_qubits for
angle encoding). Trains on statevector simulation (lightning.qubit, analytic
adjoint gradients). The ablation sweeps qubits/depth/entanglement/encoding; the
main run also trains a parameter-matched MLP head on the SAME features so
Q-PlasmaNet vs the classical head is an explicit, fair comparison.

  python -m src.train.train_quantum --config configs/quantum.yaml          # ablation + main
  python -m src.train.train_quantum --config configs/quantum.yaml --smoke  # 4q/1layer, 1 seed
  python -m src.train.train_quantum --config configs/quantum.yaml --joint  # end-to-end (slow)

--joint (end-to-end backbone+head on images) is intentionally not part of the
default scaffold; it raises with guidance so it is enabled deliberately.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import make_dataloader  # noqa: E402
from src.eval.calibration import compute_calibration  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402
from src.features.precompute import load_features  # noqa: E402
from src.models.backbones import build_backbone  # noqa: E402
from src.models.classical_heads import count_params, mlp_for_param_budget  # noqa: E402
from src.models.quantum_head import (  # noqa: E402
    build_quantum_head, build_reupload_head, count_quantum_params,
    count_reupload_params, reupload_input_dim,
)
from src.train.engine import get_device, predict, train_loop  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402

METRIC_KEYS = ["acc", "f1", "auc", "pr_auc", "balanced_acc", "ece", "brier"]


def _feature_loaders(cfg, backbone, dim, seed, bs):
    loaders = {}
    for sp in ("train", "val", "test"):
        X, y = load_features(cfg, backbone, dim, seed, sp)
        if X is None:
            loaders[sp] = None
            continue
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        loaders[sp] = DataLoader(ds, batch_size=bs, shuffle=(sp == "train"))
    return loaders


def _evaluate(model, loaders, device) -> dict:
    eval_dl = loaders["test"] or loaders["val"] or loaders["train"]
    y_true, y_prob = predict(model, eval_dl, device)
    m = compute_metrics(y_true, y_prob)
    cal = compute_calibration(y_true, y_prob)
    return {**m, "ece": cal["ece"], "brier": cal["brier"]}


def train_quantum_config(cfg, qubits, depth, entanglement, encoding, seed, backbone,
                         device_name, diff_method, epochs, lr, bs, device,
                         save_ckpt: Path | None = None) -> dict:
    if encoding == "reupload":
        feat_dim = reupload_input_dim(qubits)          # 3*qubits features per upload
        head, info = build_reupload_head(qubits, depth, entanglement,
                                         device_name=device_name, diff_method=diff_method)
    else:
        feat_dim = qubits
        head, info = build_quantum_head(qubits, depth, entanglement, encoding,
                                        device_name=device_name, diff_method=diff_method)
    loaders = _feature_loaders(cfg, backbone, feat_dim, seed, bs)
    if loaders["train"] is None:
        raise FileNotFoundError(
            f"no d{feat_dim} features for seed {seed}; run src.features.precompute --dims {feat_dim} first."
        )
    best_state, _ = train_loop(head, loaders["train"], loaders["val"], epochs=epochs, lr=lr,
                               seed=seed, device=device, verbose=False)
    head.load_state_dict(best_state)
    res = _evaluate(head, loaders, device)
    if save_ckpt is not None:
        save_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "hparams": {
            "qubits": qubits, "depth": depth, "entanglement": entanglement,
            "encoding": encoding, "device": info["device"], "diff_method": info["diff_method"],
        }, "backbone": backbone, "seed": seed}, save_ckpt)
    return {
        "model": "q_plasmanet", "qubits": qubits, "depth": depth,
        "entanglement": entanglement, "encoding": encoding, "seed": seed,
        "backbone": backbone, "device": info["device"], "diff_method": info["diff_method"],
        "params": info["params_total"], **res,
    }


def train_mlp_matched(cfg, dim, seed, backbone, target_params, epochs, lr, bs, device) -> dict:
    loaders = _feature_loaders(cfg, backbone, dim, seed, bs)
    model = mlp_for_param_budget(dim, target_params=target_params, n_classes=2)
    n = count_params(model)
    best_state, _ = train_loop(model, loaders["train"], loaders["val"], epochs=epochs, lr=lr,
                               seed=seed, device=device, verbose=False)
    model.load_state_dict(best_state)
    res = _evaluate(model, loaders, device)
    return {"model": "mlp_param_matched", "qubits": dim, "depth": np.nan,
            "entanglement": "-", "encoding": "-", "seed": seed, "backbone": backbone,
            "device": "cpu", "diff_method": "-", "params": n, **res}


def _aggregate(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = []
    for keys, g in raw.groupby(group_cols):
        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec["n_seeds"] = len(g)
        rec["params"] = int(g["params"].iloc[0])
        for k in METRIC_KEYS:
            rec[f"{k}_mean"] = float(np.nanmean(g[k]))
            rec[f"{k}_std"] = float(np.nanstd(g[k], ddof=1)) if len(g) > 1 else 0.0
        out.append(rec)
    return pd.DataFrame(out)


def run_quantum_ablation(cfg, seeds=None, smoke=False, device=None) -> pd.DataFrame:
    repo_root = Path(cfg._repo_root)
    results_dir = repo_root / cfg.get_path("paths.results")
    device = device or torch.device("cpu")  # quantum: lightning.qubit is CPU (MPS breaks TorchLayer)
    backbone = cfg.get_path("feature_backbone") or "resnet18"
    device_name = cfg.get_path("device") or "lightning.qubit"
    diff_method = cfg.get_path("diff_method") or "adjoint"
    epochs = 1 if smoke else int(cfg.get_path("train.epochs", 40) or 40)
    lr = float(cfg.get_path("train.lr", 0.01) or 0.01)
    bs = int(cfg.get_path("train.batch_size", 32) or 32)

    if smoke:
        grid = {"qubits": [4], "depth": [1], "entanglement": ["linear"], "encoding": ["angle"]}
        seeds = seeds or cfg.get_path("smoke.seeds")
    else:
        ab = cfg.get_path("ablation")
        grid = {k: ab[k] for k in ("qubits", "depth", "entanglement", "encoding")}
        seeds = seeds or cfg.get_path("seeds")

    rows = []
    combos = list(itertools.product(grid["qubits"], grid["depth"], grid["entanglement"], grid["encoding"]))
    for (q, d, ent, enc) in combos:
        for seed in seeds:
            print(f"[quantum] q={q} depth={d} ent={ent} enc={enc} seed={seed}")
            rows.append(train_quantum_config(cfg, q, d, ent, enc, seed, backbone,
                                             device_name, diff_method, epochs, lr, bs, device))
    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "quantum_ablation_raw.csv", index=False)
    agg = _aggregate(raw, ["qubits", "depth", "entanglement", "encoding"])
    agg = agg.sort_values("f1_mean", ascending=False)
    agg.to_csv(results_dir / "quantum_ablation.csv", index=False)
    print("\n[quantum] ablation (mean over seeds, top rows):")
    print(agg[["qubits", "depth", "entanglement", "encoding", "params", "f1_mean", "auc_mean"]].head(10).to_string(index=False))
    print(f"[quantum] wrote {results_dir/'quantum_ablation.csv'}")
    return agg


def run_quantum_main(cfg, seeds=None, smoke=False, device=None) -> pd.DataFrame:
    repo_root = Path(cfg._repo_root)
    results_dir = repo_root / cfg.get_path("paths.results")
    device = device or torch.device("cpu")  # quantum: lightning.qubit is CPU (MPS breaks TorchLayer)
    backbone = cfg.get_path("feature_backbone") or "resnet18"
    device_name = cfg.get_path("device") or "lightning.qubit"
    diff_method = cfg.get_path("diff_method") or "adjoint"
    epochs = 1 if smoke else int(cfg.get_path("train.epochs", 40) or 40)
    lr = float(cfg.get_path("train.lr", 0.01) or 0.01)
    bs = int(cfg.get_path("train.batch_size", 32) or 32)

    main = cfg.get_path("main") or {"qubits": 4, "depth": 1, "entanglement": "linear", "encoding": "angle"}
    if smoke:
        main = {"qubits": 4, "depth": 1, "entanglement": "linear", "encoding": "angle"}
        seeds = seeds or cfg.get_path("smoke.seeds")
    else:
        seeds = seeds or cfg.get_path("seeds")

    q, d, ent, enc = main["qubits"], main["depth"], main["entanglement"], main["encoding"]
    if enc == "reupload":
        target = count_reupload_params(q, d)["total"]
        mlp_dim = reupload_input_dim(q)                 # matched MLP sees the same features
    else:
        target = count_quantum_params(q, d)["total"]
        mlp_dim = q
    ckpt_dir = repo_root / cfg.get_path("paths.checkpoints")

    rows = []
    for seed in seeds:
        ckpt = ckpt_dir / f"quantum_main_{backbone}_seed{seed}.pt"
        rows.append(train_quantum_config(cfg, q, d, ent, enc, seed, backbone,
                                         device_name, diff_method, epochs, lr, bs, device,
                                         save_ckpt=ckpt))
        rows.append(train_mlp_matched(cfg, mlp_dim, seed, backbone, target, epochs, lr, bs, device))

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "quantum_main_raw.csv", index=False)
    agg = _aggregate(raw, ["model"])
    agg.to_csv(results_dir / "quantum_main.csv", index=False)

    print("\n[quantum] MAIN — Q-PlasmaNet vs parameter-matched MLP head "
          f"(q={q} depth={d} ent={ent} enc={enc}, on the same {mlp_dim}-dim features):")
    cols = ["model", "params", "f1_mean", "auc_mean", "acc_mean", "ece_mean"]
    print(agg[cols].to_string(index=False))
    qf = agg.set_index("model").loc["q_plasmanet", "f1_mean"]
    mf = agg.set_index("model").loc["mlp_param_matched", "f1_mean"]
    print(f"[quantum] ΔF1 (Q − MLP) = {qf - mf:+.4f}  "
          f"(paired significance test deferred to Phase 9)")
    print(f"[quantum] wrote {results_dir/'quantum_main.csv'}")
    return agg


class JointQPlasmaNet(nn.Module):
    """End-to-end: trainable CNN backbone -> learnable bottleneck -> re-uploading VQC.
    Trains on images (not cached features). Entirely on CPU (lightning head). Slow."""

    def __init__(self, n_qubits=8, depth=3, entanglement="circular"):
        super().__init__()
        self.backbone, feat = build_backbone("cnn", None, pretrained=False)
        self.bottleneck = nn.Sequential(nn.Linear(feat, 3 * n_qubits), nn.Tanh())
        self.head, _ = build_reupload_head(n_qubits, depth, entanglement,
                                           device_name="lightning.qubit",
                                           diff_method="adjoint", verify=False)

    def forward(self, x):
        z = self.bottleneck(self.backbone(x)) * math.pi      # angle range [-pi, pi]
        return self.head(z)


def run_quantum_joint(cfg, seed=0, smoke=False, limit=None) -> pd.DataFrame:
    """--joint demonstration: end-to-end backbone+VQC on images. Bounded for
    tractability (CPU + per-sample quantum sim is 10-100x slower than the frozen
    path); reports a real number, not the headline result."""
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results")
    splits = repo / cfg.get_path("paths.splits")
    device = torch.device("cpu")                              # lightning head -> CPU
    epochs = 1 if smoke else 8
    limit = limit if limit is not None else (cfg.get_path("smoke.max_samples", 200) if smoke else 400)
    bs = 32
    tr, va, te = (splits / f"seed{seed}_{s}.csv" for s in ("train", "val", "test"))

    print(f"[joint] WARNING: end-to-end backbone+VQC on images is slow "
          f"(limit={limit}, epochs={epochs}). Demonstration, not the main result.", flush=True)
    train_dl = make_dataloader(tr, cfg, train=True, batch_size=bs, limit=limit)
    val_dl = make_dataloader(va, cfg, train=False, batch_size=bs, limit=limit) if va.is_file() else None
    test_dl = make_dataloader(te, cfg, train=False, batch_size=bs, limit=limit) if te.is_file() else None

    model = JointQPlasmaNet()
    best, _ = train_loop(model, train_dl, val_dl, epochs=epochs, lr=0.01, seed=seed,
                         device=device, verbose=True)
    model.load_state_dict(best)
    y, p = predict(model, test_dl or val_dl or train_dl, device)
    m = compute_metrics(y, p)
    row = {"model": "q_plasmanet_joint", "seed": seed, "trainable_params": count_params(model),
           "limit": limit, "epochs": epochs, **m}
    pd.DataFrame([row]).to_csv(results / "quantum_joint.csv", index=False)
    print(f"[joint] test F1={m['f1']:.3f} AUC={m['auc']:.3f} acc={m['acc']:.3f} "
          f"(trainable_params={row['trainable_params']}) -> results/quantum_joint.csv", flush=True)
    return pd.DataFrame([row])


def main() -> None:
    p = argparse.ArgumentParser(description="Train the quantum head (Phase 4).")
    add_common_args(p)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--joint", action="store_true", help="End-to-end backbone+head on images (slow).")
    p.add_argument("--ablation-only", action="store_true")
    p.add_argument("--main-only", action="store_true")
    args = p.parse_args()

    configs = args.config or ["configs/backbones.yaml", "configs/quantum.yaml"]
    cfg = load_config(configs)

    if args.joint:
        # End-to-end backbone+VQC on images (bounded demo; 10-100x slower than frozen path).
        seeds = args.seeds or [0]
        for s in seeds:
            run_quantum_joint(cfg, seed=s, smoke=args.smoke)
        return

    if not args.main_only:
        run_quantum_ablation(cfg, seeds=args.seeds, smoke=args.smoke)
    if not args.ablation_only:
        run_quantum_main(cfg, seeds=args.seeds, smoke=args.smoke)


if __name__ == "__main__":
    main()
