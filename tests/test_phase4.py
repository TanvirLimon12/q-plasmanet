"""Phase-4 smoke: precompute features -> train quantum head (4q/1layer) -> ablation + main.

Uses the custom CNN backbone (no pretrained download) and a single 4-qubit config
so it stays under the ~2 min smoke budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.features.precompute import precompute_features  # noqa: E402
from src.data.splits import run_splits  # noqa: E402
from src.train.train_quantum import run_quantum_ablation, run_quantum_main  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _img(path: Path, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)


def _build(root: Path) -> None:
    rng = np.random.default_rng(0)
    for i in range(40):
        _img(root / "labeled" / "plasma" / f"p{i}.png", rng)
        _img(root / "labeled" / "non_plasma" / f"n{i}.png", rng)


def test_quantum_smoke(tmp_path):
    raw = tmp_path / "raw"
    _build(raw)

    cfg = load_config(["configs/backbones.yaml", "configs/quantum.yaml"])
    cfg["_repo_root"] = str(tmp_path)
    cfg["feature_backbone"] = "cnn"          # avoid pretrained download

    run_splits(cfg, seeds=[0], data_root=raw, strict=True)
    man = precompute_features(cfg, backbones=["cnn"], dims=[4], seeds=[0], smoke=True, data_root=raw)
    assert not man.empty
    assert (tmp_path / "data" / "processed" / "feat_labeled_cnn_d4_seed0_train.npy").is_file()

    ab = run_quantum_ablation(cfg, seeds=[0], smoke=True)
    assert (tmp_path / "results" / "quantum_ablation.csv").is_file()
    assert (ab["qubits"] == 4).all()

    mn = run_quantum_main(cfg, seeds=[0], smoke=True)
    assert set(mn["model"]) == {"q_plasmanet", "mlp_param_matched"}
    assert (tmp_path / "results" / "quantum_main.csv").is_file()

    raw_main = pd.read_csv(tmp_path / "results" / "quantum_main_raw.csv")
    for col in ("acc", "f1", "balanced_acc", "ece", "brier"):
        assert np.isfinite(raw_main[col]).all(), f"{col} non-finite"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_quantum_smoke(Path(d))
    print("OK: phase 4 smoke passed")
