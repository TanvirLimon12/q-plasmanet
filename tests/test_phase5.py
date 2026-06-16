"""Phase-5 smoke: corrupt the test split, evaluate trained classical + Q-PlasmaNet."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.robustness import run_robustness  # noqa: E402
from src.features.precompute import precompute_features  # noqa: E402
from src.data.splits import run_splits  # noqa: E402
from src.train.train_classical import run_classical  # noqa: E402
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


def test_robustness_smoke(tmp_path):
    raw = tmp_path / "raw"
    _build(raw)

    cfg = load_config(["configs/backbones.yaml", "configs/quantum.yaml"])
    cfg["_repo_root"] = str(tmp_path)
    cfg["feature_backbone"] = "cnn"

    run_splits(cfg, seeds=[0], data_root=raw, strict=True)
    run_classical(cfg, models=["cnn"], seeds=[0], smoke=True)
    precompute_features(cfg, backbones=["cnn"], dims=[4], seeds=[0], smoke=True, data_root=raw)
    run_quantum_main(cfg, seeds=[0], smoke=True)

    agg = run_robustness(
        cfg,
        model_specs=[("classical", "cnn"), ("q_plasmanet", "q_plasmanet")],
        seeds=[0], corruptions=["gaussian_noise", "blur"], severities=[2, 4], smoke=True,
    )

    assert (tmp_path / "results" / "robustness.csv").is_file()
    assert (tmp_path / "results" / "robustness_score.csv").is_file()
    assert set(agg["model"]) == {"cnn", "q_plasmanet"}

    raw_df = pd.read_csv(tmp_path / "results" / "robustness_raw.csv")
    assert (raw_df["corruption"] == "clean").any(), "clean baseline row missing"
    assert np.isfinite(raw_df["acc"]).all(), "non-finite accuracy"
    scores = pd.read_csv(tmp_path / "results" / "robustness_score.csv")
    assert {"retained_acc_ratio", "mean_acc_drop"} <= set(scores.columns)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_robustness_smoke(Path(d))
    print("OK: phase 5 smoke passed")
