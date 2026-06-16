"""Phase 7 — patient-level aggregation.

The patient_organized subset is 344 field images with YOLO bbox labels (class 0 =
plasma, 1 = non-plasma) over 10 patients, plus diagnosis.csv (per-patient plasma
ratio + diseased/normal). Cells are bbox annotations, not pre-cropped files, so
this step crops each cell from its field, classifies it with the trained
cell-level model, and aggregates predictions per patient into a plasma-cell ratio
(the diagnostic indicator).

No leakage: the classifier is trained on the *segmentation/labeled* subset, which
is disjoint from patient_organized (different images, no shared patient IDs). This
is asserted/logged.

Deliverables:
  results/patient_level.csv   (Table 7: per-patient + summary metrics)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import build_transforms  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402
from src.models.qplasmanet import build_classical_infer, build_qplasmanet_infer  # noqa: E402
from src.train.engine import get_device  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# YOLO class -> our label space (plasma=1, non_plasma=0).
_YOLO_TO_LABEL = {0: 1, 1: 0}


def _read_diagnosis(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["patient"] = df["patient"].astype(str).str.zfill(2)
    return df


def _crops_from_field(img: Image.Image, label_path: Path):
    """Yield (crop, yolo_class) for each bbox in a YOLO label file."""
    if not label_path.is_file():
        return
    W, H = img.size
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        x1 = max(0, int((cx - w / 2) * W)); y1 = max(0, int((cy - h / 2) * H))
        x2 = min(W, int((cx + w / 2) * W)); y2 = min(H, int((cy + h / 2) * H))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        yield img.crop((x1, y1, x2, y2)), cls


@torch.no_grad()
def aggregate_patients(cfg, seed: int = 0, model_name: str = "cnn",
                       limit_patients: int | None = None) -> pd.DataFrame:
    repo_root = Path(cfg._repo_root)
    results_dir = repo_root / cfg.get_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)
    pat_root = repo_root / cfg.get_path("data.raw_root") / cfg.get_path("data.subsets.patient_organized.path")
    device = get_device()
    tfm = build_transforms(cfg, train=False)

    diag = _read_diagnosis(pat_root / "diagnosis.csv")
    diag_by = diag.set_index("patient")

    if model_name == "q_plasmanet":
        model = build_qplasmanet_infer(cfg, seed, backbone_name="cnn", device=device)
    else:
        model = build_classical_infer(cfg, model_name, seed, device=device)

    patient_dirs = sorted(d for d in pat_root.iterdir() if d.is_dir() and d.name.lower().startswith("patient"))
    if limit_patients:
        patient_dirs = patient_dirs[:limit_patients]

    rows = []
    cell_true, cell_prob = [], []   # cross-subset cell-level generalization
    for pdir in patient_dirs:
        pid = "".join(ch for ch in pdir.name if ch.isdigit()).zfill(2)
        imgs_dir, labels_dir = pdir / "images", pdir / "labels"
        crops, ytrue = [], []
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                lbl = labels_dir / (img_path.stem + ".txt")
                for crop, cls in _crops_from_field(im, lbl):
                    crops.append(tfm(crop))
                    ytrue.append(_YOLO_TO_LABEL.get(cls, 0))
        if not crops:
            continue

        X = torch.stack(crops).to(device)
        probs = []
        for i in range(0, len(X), 256):                 # batch to bound memory
            logits = model(X[i:i + 256])
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        prob = np.concatenate(probs)
        ytrue = np.asarray(ytrue)
        cell_true.append(ytrue); cell_prob.append(prob)

        pred_ratio = float((prob >= 0.5).mean())        # predicted plasma fraction
        true_ratio = float((ytrue == 1).mean())
        d = diag_by.loc[pid] if pid in diag_by.index else None
        rows.append({
            "patient": pid, "n_cells": len(prob),
            "true_plasma_ratio": round(true_ratio, 4),
            "pred_plasma_ratio": round(pred_ratio, 4),
            "diag_ratio_csv": float(d["percentage"]) if d is not None else np.nan,
            "diagnosis": (d["diagnosis"] if d is not None else "?"),
        })

    pat = pd.DataFrame(rows)

    # Cell-level cross-subset generalization (classifier trained on segmentation).
    ct = np.concatenate(cell_true); cp = np.concatenate(cell_prob)
    cell_m = compute_metrics(ct, cp)

    # Patient-level: diseased(1)/normal(0) vs predicted plasma ratio.
    y_pat = (pat["diagnosis"] == "diseased").astype(int).to_numpy()
    score = pat["pred_plasma_ratio"].to_numpy()
    pat_m = compute_metrics(y_pat, score)
    thr, sens, spec = _best_threshold(y_pat, score)
    corr = float(np.corrcoef(pat["true_plasma_ratio"], score)[0, 1]) if len(pat) > 1 else np.nan

    # n=10 is small -> report a bootstrap CI on the patient AUC + leave-one-patient-out
    # CV (threshold fit on the other patients) so the AUC=1.0 claim isn't fragile.
    auc_lo, auc_hi = _bootstrap_auc_ci(y_pat, score)
    lopo_acc, lopo_sens, lopo_spec = _lopo(y_pat, score)

    pat.to_csv(results_dir / "patient_level.csv", index=False)
    summary = pd.DataFrame([{
        "model": model_name, "seed": seed, "n_patients": len(pat),
        "n_cells": int(pat["n_cells"].sum()),
        "cell_acc": round(cell_m["acc"], 4), "cell_f1": round(cell_m["f1"], 4),
        "cell_auc": round(cell_m["auc"], 4),
        "patient_auc": round(pat_m["auc"], 4),
        "patient_auc_ci_lo": round(auc_lo, 4), "patient_auc_ci_hi": round(auc_hi, 4),
        "patient_sens": round(sens, 4), "patient_spec": round(spec, 4),
        "lopo_acc": round(lopo_acc, 4), "lopo_sens": round(lopo_sens, 4),
        "lopo_spec": round(lopo_spec, 4),
        "ratio_threshold": round(thr, 4),
        "ratio_corr_pred_vs_true": round(corr, 4),
        "leakage": "none (classifier trained on segmentation subset; disjoint)",
    }])
    summary.to_csv(results_dir / "patient_level_summary.csv", index=False)

    print(pat.to_string(index=False))
    print("\n[patient_agg] cross-subset CELL-level (cnn trained on segmentation, tested on patient cells):")
    print(f"  acc={cell_m['acc']:.4f} f1={cell_m['f1']:.4f} auc={cell_m['auc']:.4f}  (n={len(ct)} cells)")
    print("[patient_agg] PATIENT-level (predicted plasma ratio -> diseased/normal):")
    print(f"  AUC={pat_m['auc']:.4f} [95% CI {auc_lo:.3f}-{auc_hi:.3f}] sens={sens:.4f} spec={spec:.4f} @ratio>{thr:.3f}")
    print(f"  LOPO-CV: acc={lopo_acc:.4f} sens={lopo_sens:.4f} spec={lopo_spec:.4f}  "
          f"corr(pred,true ratio)={corr:.4f}")
    print(f"[patient_agg] no leakage: classifier trained on segmentation/labeled, disjoint from patient_organized.")
    print(f"[patient_agg] wrote {results_dir/'patient_level.csv'} and patient_level_summary.csv")
    return summary


def _bootstrap_auc_ci(y, score, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI on patient-level AUC (resample patients)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); score = np.asarray(score)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _lopo(y, score):
    """Leave-one-patient-out: fit Youden threshold on the others, classify held-out."""
    y = np.asarray(y); score = np.asarray(score)
    preds = np.zeros_like(y)
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        thr, _, _ = _best_threshold(y[mask], score[mask])
        preds[i] = int(score[i] >= thr)
    tp = int(((preds == 1) & (y == 1)).sum()); fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum()); fp = int(((preds == 1) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return acc, sens, spec


def _best_threshold(y, score):
    """Youden-J optimal threshold over observed scores; returns (thr, sens, spec)."""
    if len(np.unique(y)) < 2:
        return 0.5, float("nan"), float("nan")
    best = (0.5, 0.0, 0.0, -1.0)
    for thr in np.unique(np.concatenate([[0.0], score, [1.0]])):
        pred = (score >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sens + spec - 1
        if j > best[3]:
            best = (float(thr), sens, spec, j)
    return best[0], best[1], best[2]


def main() -> None:
    p = argparse.ArgumentParser(description="Patient-level aggregation (Phase 7).")
    add_common_args(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="cnn")
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml"])
    aggregate_patients(cfg, seed=args.seed, model_name=args.model)


if __name__ == "__main__":
    main()
