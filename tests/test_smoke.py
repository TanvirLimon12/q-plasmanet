"""Phase-1 smoke test: audit runs on a tiny synthetic PCMMD tree without error.

Outputs are redirected to a temp dir so the real results/ deliverables are not
overwritten by synthetic data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.audit import run_audit  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _img(path: Path, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _build_synthetic_pcmmd(root: Path) -> None:
    rng = np.random.default_rng(0)
    # labeled: plasma / non_plasma
    for i in range(4):
        _img(root / "labeled" / "plasma" / f"p{i}.png", rng)
        _img(root / "labeled" / "non_plasma" / f"n{i}.png", rng)
    # patient_organized: two patients
    for pid in ("patient_001", "patient_002"):
        for i in range(3):
            _img(root / "patient_organized" / pid / f"c{i}.png", rng)
    # segmented: plasma / non_plasma
    for i in range(3):
        _img(root / "segmented" / "plasma" / f"sp{i}.png", rng)
        _img(root / "segmented" / "non_plasma" / f"sn{i}.png", rng)


def test_audit_smoke(tmp_path):
    raw = tmp_path / "raw"
    _build_synthetic_pcmmd(raw)

    cfg = load_config(None)
    cfg["_repo_root"] = str(tmp_path)          # redirect results/ to temp

    df = run_audit(cfg, smoke=True, data_root=raw)

    assert len(df) == 3, "expected three subsets"
    assert df["path_exists"].all(), "all synthetic subsets should exist"
    assert df["n_images"].sum() > 0

    labeled = df.set_index("subset").loc["labeled"]
    assert labeled["n_plasma"] == 4 and labeled["n_non_plasma"] == 4

    pat = df.set_index("subset").loc["patient_organized"]
    assert pat["n_patients"] == 2, "two patients expected"
    assert bool(pat["has_patient_ids"]) is True

    assert (tmp_path / "results" / "dataset_summary.csv").is_file()
    assert (tmp_path / "results" / "audit_log.txt").is_file()


def test_missing_subset_is_reported_not_crashed(tmp_path):
    # Only build one subset; the other two must be reported absent, not crash.
    raw = tmp_path / "raw"
    rng = np.random.default_rng(1)
    for i in range(3):
        _img(raw / "labeled" / "plasma" / f"p{i}.png", rng)

    cfg = load_config(None)
    cfg["_repo_root"] = str(tmp_path)
    df = run_audit(cfg, smoke=True, data_root=raw)

    by = df.set_index("subset")
    assert bool(by.loc["labeled"]["path_exists"]) is True
    assert bool(by.loc["segmented"]["path_exists"]) is False
    assert by.loc["segmented"]["n_images"] == 0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_audit_smoke(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_missing_subset_is_reported_not_crashed(Path(d))
    print("OK: smoke tests passed")
