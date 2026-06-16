"""Phase 4 — precompute frozen-backbone features.

Freeze the chosen backbone, run each split through once, compress to D-dim
vectors (D in {4,6,8} = qubit counts for angle encoding), and dump to .npy. The
quantum head then trains on these cached vectors instead of images — the single
biggest speedup.

Compression: PCA fit on the TRAIN split only (no leakage), then MinMax-scaled to
[-pi, pi] so the values are usable as angle-encoding rotations. The fitted PCA +
scaler are applied unchanged to val/test.

Outputs (per backbone, D, seed, split):
  data/processed/feat_{subset}_{backbone}_d{D}_seed{seed}_{split}.npy   (X: N×D)
  data/processed/lab_{subset}_{backbone}_d{D}_seed{seed}_{split}.npy    (y: N)
  results/feature_manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import make_dataloader  # noqa: E402
from src.models.backbones import build_backbone, freeze  # noqa: E402
from src.train.engine import get_device  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402

_PI = float(np.pi)


def _split_csv(splits_dir: Path, subset: str, seed: int, split: str) -> Path:
    return splits_dir / f"seed{seed}_{split}.csv"   # primary subset only


@torch.no_grad()
def extract_features(backbone, loader, device) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    feats, labs = [], []
    for x, y in loader:
        x = x.to(device)
        f = backbone(x)
        feats.append(f.cpu().numpy())
        labs.append(np.asarray(y))
    if not feats:
        return np.zeros((0, 0)), np.zeros((0,))
    return np.concatenate(feats), np.concatenate(labs)


def precompute_features(cfg, backbones=None, dims=None, seeds=None, smoke: bool = False,
                        data_root: Path | None = None, from_classifier: str | None = None) -> pd.DataFrame:
    """If ``from_classifier`` is a Phase-3 model name (e.g. 'cnn'), the backbone is
    loaded with that model's PCMMD-TRAINED weights (per seed) instead of frozen
    ImageNet/random weights — giving the quantum head task-relevant features."""
    repo_root = Path(cfg._repo_root)
    splits_dir = repo_root / cfg.get_path("paths.splits")
    out_dir = repo_root / cfg.get_path("data.processed_root")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = repo_root / cfg.get_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)
    subset = cfg.get_path("data.primary_subset")
    device = get_device()

    backbones = backbones or [cfg.get_path("feature_backbone") or "resnet18"]
    dims = dims or cfg.get_path("feature_dims") or [4, 6, 8]
    if smoke:
        seeds = seeds or cfg.get_path("smoke.seeds")
        dims = [min(dims)]
        pretrained = False
        limit = cfg.get_path("smoke.max_samples", 200)
    else:
        seeds = seeds or cfg.get_path("seeds")
        pretrained = True
        limit = None

    bs = int(cfg.get_path("backbones.train.batch_size", 64) or 64)
    splits = ["train", "val", "test"]
    manifest = []

    ckpt_dir = repo_root / cfg.get_path("paths.checkpoints")
    for bname in backbones:
        backbone, feat_dim = build_backbone(bname, cfg, pretrained=pretrained)
        freeze(backbone)
        backbone.to(device)
        # Persist exact frozen weights so later phases reload an identical backbone
        # (custom CNN is randomly initialized; pretrained weights are fixed but we
        # save anyway for a single source of truth).
        bfile = out_dir / f"backbone_{subset}_{bname}.pt"
        torch.save(backbone.state_dict(), bfile)

        for seed in seeds:
            if from_classifier:
                # Load PCMMD-trained backbone weights for this seed from a Phase-3 ckpt.
                ck = ckpt_dir / f"{subset}_{from_classifier}_seed{seed}.pt"
                if not ck.is_file():
                    raise FileNotFoundError(f"--from-classifier {from_classifier}: missing {ck}")
                sd = torch.load(ck, map_location=device, weights_only=False)["state_dict"]
                bsd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
                backbone.load_state_dict(bsd)
                freeze(backbone)
                backbone.to(device)
                torch.save(backbone.state_dict(), out_dir / f"backbone_{subset}_{bname}_seed{seed}.pt")
            # Raw backbone features per split.
            raw = {}
            for sp in splits:
                csv = _split_csv(splits_dir, subset, seed, sp)
                if not csv.is_file():
                    continue
                dl = make_dataloader(csv, cfg, train=False, batch_size=bs, limit=limit, shuffle=False)
                raw[sp] = extract_features(backbone, dl, device)

            if "train" not in raw or raw["train"][0].shape[0] == 0:
                raise FileNotFoundError(
                    f"no train features for seed {seed}; run Phase 2 splits first."
                )

            for D in dims:
                Xtr, ytr = raw["train"]
                n_comp = min(D, Xtr.shape[0], Xtr.shape[1])
                # Fit on TRAIN only; persist the fitted transform so the VQC pipeline
                # reproduces on new/corrupted images (Phases 5-6).
                transform = Pipeline([
                    ("pca", PCA(n_components=n_comp, random_state=seed)),
                    ("scale", MinMaxScaler(feature_range=(-_PI, _PI))),
                ]).fit(Xtr)
                pca = transform.named_steps["pca"]
                tfile = out_dir / f"transform_{subset}_{bname}_d{D}_seed{seed}.joblib"
                joblib.dump(transform, tfile)

                for sp in splits:
                    if sp not in raw:
                        continue
                    X, y = raw[sp]
                    Z = transform.transform(X)
                    if Z.shape[1] < D:                # pad if PCA gave fewer comps
                        Z = np.pad(Z, ((0, 0), (0, D - Z.shape[1])))
                    fx = out_dir / f"feat_{subset}_{bname}_d{D}_seed{seed}_{sp}.npy"
                    fy = out_dir / f"lab_{subset}_{bname}_d{D}_seed{seed}_{sp}.npy"
                    np.save(fx, Z.astype(np.float32))
                    np.save(fy, y.astype(np.int64))
                    manifest.append({
                        "backbone": bname, "dim": D, "seed": seed, "split": sp,
                        "n": int(Z.shape[0]), "feat_file": fx.name, "lab_file": fy.name,
                        "explained_var": float(pca.explained_variance_ratio_.sum()),
                    })

    df = pd.DataFrame(manifest)
    df.to_csv(results_dir / "feature_manifest.csv", index=False)
    print(df.to_string(index=False) if not df.empty else "[precompute] no features written")
    print(f"\n[precompute] features in {out_dir}")
    print(f"[precompute] wrote {results_dir/'feature_manifest.csv'}")
    return df


def load_features(cfg, backbone: str, dim: int, seed: int, split: str):
    repo_root = Path(cfg._repo_root)
    out_dir = repo_root / cfg.get_path("data.processed_root")
    subset = cfg.get_path("data.primary_subset")
    fx = out_dir / f"feat_{subset}_{backbone}_d{dim}_seed{seed}_{split}.npy"
    fy = out_dir / f"lab_{subset}_{backbone}_d{dim}_seed{seed}_{split}.npy"
    if not fx.is_file():
        return None, None
    return np.load(fx), np.load(fy)


def load_transform(cfg, backbone: str, dim: int, seed: int):
    """Load the fitted PCA+scaler Pipeline for (backbone, dim, seed), or None."""
    repo_root = Path(cfg._repo_root)
    out_dir = repo_root / cfg.get_path("data.processed_root")
    subset = cfg.get_path("data.primary_subset")
    tfile = out_dir / f"transform_{subset}_{backbone}_d{dim}_seed{seed}.joblib"
    if not tfile.is_file():
        return None
    return joblib.load(tfile)


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute frozen-backbone features (Phase 4).")
    add_common_args(p)
    p.add_argument("--backbones", nargs="*", default=None)
    p.add_argument("--dims", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--from-classifier", default=None,
                   help="Phase-3 model name (e.g. cnn) to source PCMMD-trained backbone weights.")
    args = p.parse_args()
    configs = args.config or ["configs/backbones.yaml", "configs/quantum.yaml"]
    cfg = load_config(configs)
    precompute_features(cfg, backbones=args.backbones, dims=args.dims, seeds=args.seeds,
                        smoke=args.smoke, data_root=Path(args.data_root) if args.data_root else None,
                        from_classifier=args.from_classifier)


if __name__ == "__main__":
    main()
