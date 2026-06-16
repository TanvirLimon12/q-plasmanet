"""Phase-6 smoke: re-evaluate trained VQC under 128 shots + depolarizing noise."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.nisq import run_nisq  # noqa: E402
from src.features.precompute import precompute_features  # noqa: E402
from src.data.splits import run_splits  # noqa: E402
from src.train.train_quantum import run_quantum_main  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _img(path: Path, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)


def _build(root: Path) -> None:
    rng = np.random.default_rng(0)
    for i in range(40):
        _img(root / "labeled" / "plasma" / f"p{i}.png", rng)
        _img(root / "labeled" / "non_plasma" / f"n{i}.png", rng)


def test_nisq_smoke(tmp_path):
    raw = tmp_path / "raw"
    _build(raw)

    cfg = load_config(["configs/backbones.yaml", "configs/quantum.yaml", "configs/noise.yaml"])
    cfg["_repo_root"] = str(tmp_path)
    cfg["feature_backbone"] = "cnn"

    run_splits(cfg, seeds=[0], data_root=raw, strict=True)
    precompute_features(cfg, backbones=["cnn"], dims=[4], seeds=[0], smoke=True, data_root=raw)
    run_quantum_main(cfg, seeds=[0], smoke=True)

    df = run_nisq(cfg, seeds=[0], smoke=True, limit=16)

    assert (tmp_path / "results" / "nisq.csv").is_file()
    assert {"shots", "noise"} <= set(df["study"])          # 'calibrated' also present
    assert (df["study"] == "shots").any() and (df[df.study == "shots"]["shots"] == 128).any()
    assert (df[df.study == "noise"]["noise_type"] == "depolarizing").any()
    assert (df["study"] == "calibrated").any()             # calibration-informed noise study
    assert np.isfinite(df["acc"]).all(), "non-finite accuracy under noise"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_nisq_smoke(Path(d))
    print("OK: phase 6 smoke passed")
