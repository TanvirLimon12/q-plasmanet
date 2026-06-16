"""Phase-7 smoke: crop patient bboxes, classify, aggregate per patient — no crash,
finite metrics, leakage logged."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.splits import run_splits  # noqa: E402
from src.eval.patient_agg import aggregate_patients  # noqa: E402
from src.train.train_classical import run_classical  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _img(path: Path, rng, hw=(64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (*hw, 3), dtype=np.uint8)).save(path)


def _label(path: Path, n_plasma, n_non, rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, n in ((0, n_plasma), (1, n_non)):       # 0=plasma, 1=non
        for _ in range(n):
            cx, cy = rng.uniform(0.2, 0.8, 2)
            lines.append(f"{cls} {cx:.3f} {cy:.3f} 0.15 0.15")
    path.write_text("\n".join(lines) + "\n")


def _build(root: Path):
    rng = np.random.default_rng(0)
    # labeled cells (train the cnn classifier)
    for i in range(30):
        _img(root / "labeled" / "plasma" / f"p{i}.png", rng, (16, 16))
        _img(root / "labeled" / "non_plasma" / f"n{i}.png", rng, (16, 16))
    # patient_organized: 2 diseased (more plasma) + 2 normal (few plasma)
    pat = root / "patient_organized"
    diag = []
    for pid, (npl, nnon, dx) in {
        "01": (8, 4, "diseased"), "02": (7, 5, "diseased"),
        "03": (1, 11, "normal"), "04": (2, 10, "normal"),
    }.items():
        _img(pat / f"patient {pid}" / "images" / "f0.jpg", rng, (128, 128))
        _label(pat / f"patient {pid}" / "labels" / "f0.txt", npl, nnon, rng)
        diag.append({"patient": pid, "plasma_cells": npl, " non_plasma_cells": nnon,
                     "total": npl + nnon, "percentage": npl / (npl + nnon), "diagnosis": dx})
    pd.DataFrame(diag).to_csv(pat / "diagnosis.csv", index=False)


def test_patient_agg_smoke(tmp_path):
    raw = tmp_path / "data" / "raw" / "pcmmd"
    _build(raw)
    cfg = load_config(["configs/backbones.yaml"])
    cfg["_repo_root"] = str(tmp_path)

    run_splits(cfg, seeds=[0], data_root=raw, strict=True)
    run_classical(cfg, models=["cnn"], seeds=[0], smoke=True)

    summary = aggregate_patients(cfg, seed=0, model_name="cnn")

    assert (tmp_path / "results" / "patient_level.csv").is_file()
    pl = pd.read_csv(tmp_path / "results" / "patient_level.csv")
    assert len(pl) == 4 and np.isfinite(pl["pred_plasma_ratio"]).all()
    assert summary.iloc[0]["n_patients"] == 4
    assert np.isfinite(summary.iloc[0]["cell_auc"])
    assert "none" in str(summary.iloc[0]["leakage"])


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_patient_agg_smoke(Path(d))
    print("OK: phase 7 smoke passed")
