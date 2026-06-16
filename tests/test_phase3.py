"""Phase-3 smoke: classical baselines train 1 epoch on a tiny subset, metrics print."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.splits import run_splits  # noqa: E402
from src.train.train_classical import run_classical  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _img(path: Path, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)


def _build(root: Path) -> None:
    rng = np.random.default_rng(0)
    for i in range(30):
        _img(root / "labeled" / "plasma" / f"p{i}.png", rng)
        _img(root / "labeled" / "non_plasma" / f"n{i}.png", rng)


def test_classical_smoke(tmp_path):
    raw = tmp_path / "raw"
    _build(raw)

    cfg = load_config(["configs/backbones.yaml"])
    cfg["_repo_root"] = str(tmp_path)
    run_splits(cfg, seeds=[0], data_root=raw, strict=True)

    agg = run_classical(cfg, models=["cnn", "cnn_mlp"], seeds=[0], smoke=True)

    assert set(agg["model"]) == {"cnn", "cnn_mlp"}
    assert (tmp_path / "results" / "baselines.csv").is_file()
    assert (tmp_path / "results" / "baselines_raw.csv").is_file()

    raw_df = pd.read_csv(tmp_path / "results" / "baselines_raw.csv")
    for col in ("acc", "f1", "balanced_acc", "ece", "brier"):
        assert np.isfinite(raw_df[col]).all(), f"{col} has non-finite values"

    for model in ("cnn", "cnn_mlp"):
        assert (tmp_path / "checkpoints" / f"labeled_{model}_seed0.pt").is_file()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_classical_smoke(Path(d))
    print("OK: phase 3 smoke passed")
