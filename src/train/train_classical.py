"""Phase 3 — classical baselines.

Trains, over 5 seeds: custom CNN, MobileNetV2, ResNet-18, EfficientNet-B0, and
CNN + parameter-matched MLP head (the key fair comparator vs the quantum head).
Each (model, seed) is evaluated on the test split; metrics + calibration are
written to results/baselines_raw.csv, aggregated (mean ± std) to results/baselines.csv,
and the best checkpoint saved per model/seed.

  python -m src.train.train_classical --config configs/backbones.yaml
  python -m src.train.train_classical --config configs/backbones.yaml --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import make_dataloader  # noqa: E402
from src.eval.calibration import compute_calibration  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402
from src.models.classical_heads import ClassicalClassifier, build_classifier, count_params  # noqa: E402
from src.train.engine import get_device, predict, train_loop  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402

# cnn_mlp dropped: the custom-CNN + tiny-MLP-head end-to-end variant was an unstable
# placeholder (h≈1 bottleneck → seed collapse). The proper parameter-matched classical
# comparator is the MLP head on d24 features in quantum_main.csv.
DEFAULT_MODELS = ["cnn", "mobilenet_v2", "resnet18", "efficientnet_b0"]
SMOKE_MODELS = ["cnn", "cnn_mlp"]
METRIC_KEYS = ["acc", "f1", "auc", "pr_auc", "balanced_acc", "ece", "brier"]


def _split_csv(splits_dir: Path, subset: str, primary: str, seed: int, split: str) -> Path:
    prefix = "" if subset == primary else f"{subset}_"
    return splits_dir / f"{prefix}seed{seed}_{split}.csv"


def run_classical(cfg, models=None, seeds=None, smoke: bool = False,
                  data_root: Path | None = None) -> pd.DataFrame:
    repo_root = Path(cfg._repo_root)
    splits_dir = repo_root / cfg.get_path("paths.splits")
    ckpt_dir = repo_root / cfg.get_path("paths.checkpoints")
    results_dir = repo_root / cfg.get_path("paths.results")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    subset = cfg.get_path("data.primary_subset")
    device = get_device()

    if smoke:
        models = models or SMOKE_MODELS
        seeds = seeds or cfg.get_path("smoke.seeds")
        epochs = 1
        limit = cfg.get_path("smoke.max_samples", 200)
        pretrained = False
    else:
        models = models or DEFAULT_MODELS
        seeds = seeds or cfg.get_path("seeds")
        epochs = int(cfg.get_path("backbones.train.epochs", 30) or 30)
        limit = None
        pretrained = None

    lr = float(cfg.get_path("backbones.train.lr", 1e-3) or 1e-3)
    bs = int(cfg.get_path("backbones.train.batch_size", 64) or 64)
    opt = cfg.get_path("backbones.train.optimizer", "adam") or "adam"
    wd = float(cfg.get_path("backbones.train.weight_decay", 0.0) or 0.0)

    rows = []
    for model_name in models:
        for seed in seeds:
            tr = _split_csv(splits_dir, subset, subset, seed, "train")
            va = _split_csv(splits_dir, subset, subset, seed, "val")
            te = _split_csv(splits_dir, subset, subset, seed, "test")
            if not tr.is_file():
                raise FileNotFoundError(
                    f"missing split {tr} — run Phase 2 (src.data.splits) first."
                )

            train_dl = make_dataloader(tr, cfg, train=True, batch_size=bs, limit=limit)
            val_dl = make_dataloader(va, cfg, train=False, batch_size=bs, limit=limit) if va.is_file() else None
            test_dl = make_dataloader(te, cfg, train=False, batch_size=bs, limit=limit) if te.is_file() else None

            model = build_classifier(model_name, cfg, n_classes=2, pretrained=pretrained)
            n_train = count_params(model, trainable_only=True)
            print(f"\n[classical] model={model_name} seed={seed} trainable_params={n_train}")

            best_state, _hist = train_loop(
                model, train_dl, val_dl, epochs=epochs, lr=lr, seed=seed,
                device=device, optimizer=opt, weight_decay=wd, verbose=True,
            )
            model.load_state_dict(best_state)

            eval_dl = test_dl or val_dl or train_dl
            y_true, y_prob = predict(model, eval_dl, device)
            m = compute_metrics(y_true, y_prob)
            cal = compute_calibration(y_true, y_prob)

            ckpt = ckpt_dir / f"{subset}_{model_name}_seed{seed}.pt"
            torch.save({"state_dict": best_state, "model": model_name, "seed": seed}, ckpt)

            rows.append({
                "model": model_name, "seed": seed, "trainable_params": n_train,
                "acc": m["acc"], "f1": m["f1"], "auc": m["auc"],
                "pr_auc": m["pr_auc"], "balanced_acc": m["balanced_acc"],
                "ece": cal["ece"], "brier": cal["brier"],
            })

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "baselines_raw.csv", index=False)

    # Aggregate mean ± std over seeds.
    agg_rows = []
    for model_name, g in raw.groupby("model"):
        rec = {"model": model_name, "n_seeds": len(g),
               "trainable_params": int(g["trainable_params"].iloc[0])}
        for k in METRIC_KEYS:
            rec[f"{k}_mean"] = float(np.nanmean(g[k]))
            rec[f"{k}_std"] = float(np.nanstd(g[k], ddof=1)) if len(g) > 1 else 0.0
        agg_rows.append(rec)
    agg = pd.DataFrame(agg_rows).sort_values("f1_mean", ascending=False)
    agg.to_csv(results_dir / "baselines.csv", index=False)

    print("\n[classical] baselines (mean over seeds):")
    cols = ["model", "n_seeds", "trainable_params", "f1_mean", "auc_mean", "acc_mean", "ece_mean"]
    print(agg[cols].to_string(index=False))
    print(f"\n[classical] wrote {results_dir/'baselines.csv'} and baselines_raw.csv")
    if not agg.empty:
        best = agg.iloc[0]
        print(f"[classical] strongest by F1: {best['model']} "
              f"F1={best['f1_mean']:.4f} AUC={best['auc_mean']:.4f} "
              f"(this is the bar the quantum head must clear)")
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Train classical baselines (Phase 3).")
    add_common_args(p)
    p.add_argument("--models", nargs="*", default=None, help="Subset of models to train.")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    configs = args.config or ["configs/backbones.yaml"]
    cfg = load_config(configs)
    run_classical(cfg, models=args.models, seeds=args.seeds, smoke=args.smoke,
                  data_root=Path(args.data_root) if args.data_root else None)


if __name__ == "__main__":
    main()
