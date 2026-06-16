"""Phase 5 — robustness under realistic microscopy variation. Inference only.

Applies brightness, contrast, stain shift, blur, Gaussian noise, and JPEG
compression at graded severities (1-5) to the test split, and evaluates the
already-trained models (no retraining). Reports per-(model, corruption, severity)
metrics plus a corruption-robustness score per model.

Outputs:
  results/robustness_raw.csv   (per model/seed/corruption/severity)
  results/robustness.csv       (mean ± std over seeds)
  results/robustness_score.csv (retained-accuracy score per model)
"""
from __future__ import annotations

import argparse
import io
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import make_dataloader  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402
from src.models.qplasmanet import build_classical_infer, build_qplasmanet_infer  # noqa: E402
from src.train.engine import get_device, predict  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402


# --- corruption functions: callable(PIL.Image, severity 1..5) -> PIL.Image ---

def c_brightness(img, sev):
    return ImageEnhance.Brightness(img).enhance(1.0 - 0.15 * sev)   # darken


def c_contrast(img, sev):
    return ImageEnhance.Contrast(img).enhance(1.0 - 0.15 * sev)     # flatten


def c_stain(img, sev):
    hsv = np.asarray(img.convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(12 * sev)) % 256               # hue shift
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + 0.12 * sev), 0, 255) # saturation
    out = Image.new("HSV", img.size)                                 # avoid fromarray mode-deprecation
    out.putdata([tuple(p) for p in hsv.astype(np.uint8).reshape(-1, 3)])
    return out.convert("RGB")


def c_blur(img, sev):
    return img.filter(ImageFilter.GaussianBlur(radius=0.5 * sev))


def c_gaussian_noise(img, sev):
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.default_rng(0).normal(0, 0.05 * sev * 255, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def c_jpeg(img, sev):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(5, 50 - 9 * sev))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


CORRUPTIONS = {
    "brightness": c_brightness, "contrast": c_contrast, "stain": c_stain,
    "blur": c_blur, "gaussian_noise": c_gaussian_noise, "jpeg": c_jpeg,
}


def _build_model(cfg, kind: str, name: str, seed: int, device):
    if kind == "q_plasmanet":
        return build_qplasmanet_infer(cfg, seed, device=device)
    return build_classical_infer(cfg, name, seed, device=device)


def _eval(model, test_csv, cfg, bs, device, corruption=None, limit=None) -> dict:
    dl = make_dataloader(test_csv, cfg, train=False, batch_size=bs, shuffle=False,
                         corruption=corruption, limit=limit)
    y_true, y_prob = predict(model, dl, device)
    return compute_metrics(y_true, y_prob)


def run_robustness(cfg, model_specs=None, seeds=None, corruptions=None, severities=None,
                   smoke: bool = False, limit=None) -> pd.DataFrame:
    repo_root = Path(cfg._repo_root)
    splits_dir = repo_root / cfg.get_path("paths.splits")
    results_dir = repo_root / cfg.get_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Inference only; q_plasmanet uses the CPU quantum head, so evaluate all models
    # on CPU for a single consistent device (avoids MPS/CPU tensor mismatches).
    device = torch.device("cpu")
    bs = int(cfg.get_path("backbones.train.batch_size", 64) or 64)

    if smoke:
        corruptions = corruptions or ["gaussian_noise"]
        severities = severities or [3]
        seeds = seeds or cfg.get_path("smoke.seeds")
        limit = limit if limit is not None else cfg.get_path("smoke.max_samples", 200)
    else:
        corruptions = corruptions or list(CORRUPTIONS)
        severities = severities or [1, 2, 3, 4, 5]
        seeds = seeds or cfg.get_path("seeds")
    model_specs = model_specs or [("classical", "cnn")]

    rows = []
    for seed in seeds:
        test_csv = splits_dir / f"seed{seed}_test.csv"
        if not test_csv.is_file():
            raise FileNotFoundError(f"missing {test_csv}; run Phase 2 splits first.")
        for kind, name in model_specs:
            model = _build_model(cfg, kind, name, seed, device)
            clean = _eval(model, test_csv, cfg, bs, device, corruption=None, limit=limit)
            rows.append({"model": name, "seed": seed, "corruption": "clean",
                         "severity": 0, **clean})
            for cname in corruptions:
                fn = CORRUPTIONS[cname]
                for sev in severities:
                    m = _eval(model, test_csv, cfg, bs, device,
                              corruption=partial(fn, sev=sev), limit=limit)
                    rows.append({"model": name, "seed": seed, "corruption": cname,
                                 "severity": sev, **m})

    raw = pd.DataFrame(rows)
    raw.to_csv(results_dir / "robustness_raw.csv", index=False)

    # Mean ± std over seeds, per (model, corruption, severity).
    agg = (raw.groupby(["model", "corruption", "severity"])
           .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                f1_mean=("f1", "mean"), auc_mean=("auc", "mean"))
           .reset_index())
    agg.to_csv(results_dir / "robustness.csv", index=False)

    # Corruption-robustness score: mean retained accuracy (corrupt / clean).
    score_rows = []
    for model, g in raw.groupby("model"):
        clean_acc = g[g["corruption"] == "clean"]["acc"].mean()
        corrupt = g[g["corruption"] != "clean"]["acc"]
        retained = float(corrupt.mean() / clean_acc) if clean_acc > 0 else float("nan")
        score_rows.append({
            "model": model, "clean_acc": float(clean_acc),
            "mean_corrupt_acc": float(corrupt.mean()),
            "retained_acc_ratio": retained,
            "mean_acc_drop": float(clean_acc - corrupt.mean()),
        })
    scores = pd.DataFrame(score_rows).sort_values("retained_acc_ratio", ascending=False)
    scores.to_csv(results_dir / "robustness_score.csv", index=False)

    print("\n[robustness] retained-accuracy score per model:")
    print(scores.to_string(index=False))
    print(f"[robustness] wrote {results_dir/'robustness.csv'} and robustness_score.csv")
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Robustness suite (Phase 5, inference only).")
    add_common_args(p)
    p.add_argument("--models", nargs="*", default=None,
                   help="classical model names; 'q_plasmanet' for the quantum pipeline.")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    args = p.parse_args()
    configs = args.config or ["configs/backbones.yaml", "configs/quantum.yaml"]
    cfg = load_config(configs)

    specs = None
    if args.models:
        specs = [("q_plasmanet", "q_plasmanet") if m == "q_plasmanet" else ("classical", m)
                 for m in args.models]
    run_robustness(cfg, model_specs=specs, seeds=args.seeds, smoke=args.smoke)


if __name__ == "__main__":
    main()
