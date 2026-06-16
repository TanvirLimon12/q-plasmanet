"""Phase-2 smoke: splits build with zero patient leakage; dataloader yields a batch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset import make_dataloader  # noqa: E402
from src.data.splits import run_splits  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def _img(path: Path, rng) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(path)


def _build(root: Path) -> None:
    rng = np.random.default_rng(0)
    # labeled (stratified): 30 plasma + 30 non_plasma
    for i in range(30):
        _img(root / "labeled" / "plasma" / f"p{i}.png", rng)
        _img(root / "labeled" / "non_plasma" / f"n{i}.png", rng)
    # patient_organized (patient-level): 12 patients x 5 cells
    for pid in range(1, 13):
        for c in range(5):
            _img(root / "patient_organized" / f"patient_{pid:03d}" / f"c{c}.png", rng)


def test_splits_no_leakage_and_ratios(tmp_path):
    raw = tmp_path / "raw"
    _build(raw)

    cfg = load_config(None)
    cfg["_repo_root"] = str(tmp_path)
    summary = run_splits(cfg, seeds=[0, 1], data_root=raw, strict=True)

    assert int(summary["leakage"].max()) == 0, "patient leakage must be zero"

    # primary subset (labeled) uses doc-exact filenames
    for split in ("train", "val", "test"):
        assert (tmp_path / "splits" / f"seed0_{split}.csv").is_file()

    # patient_organized: verify zero patient overlap directly from the files
    sp = tmp_path / "splits"
    pats = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(sp / f"patient_organized_seed0_{split}.csv")
        pats[split] = set(df["patient_id"].unique())
    assert pats["train"] & pats["val"] == set()
    assert pats["train"] & pats["test"] == set()
    assert pats["val"] & pats["test"] == set()
    assert sum(len(v) for v in pats.values()) == 12, "all patients accounted for"

    # labeled stratified: roughly 70/15/15 and class-balanced
    lab = summary[(summary.subset == "labeled") & (summary.seed == 0)].set_index("split")
    assert lab.loc["train", "n"] == 42  # 0.70 * 60
    assert abs(lab.loc["train", "n_plasma"] - lab.loc["train", "n_non_plasma"]) <= 1

    assert (tmp_path / "results" / "split_summary.csv").is_file()


def test_dataloader_batch_shape(tmp_path):
    raw = tmp_path / "raw"
    _build(raw)
    cfg = load_config(None)
    cfg["_repo_root"] = str(tmp_path)
    run_splits(cfg, seeds=[0], data_root=raw, strict=True)

    train_csv = tmp_path / "splits" / "seed0_train.csv"
    dl = make_dataloader(train_csv, cfg, train=True, batch_size=8)
    x, y = next(iter(dl))
    size = int(cfg.get_path("image.size"))
    assert tuple(x.shape) == (8, 3, size, size), f"unexpected batch shape {tuple(x.shape)}"
    assert len(y) == 8
    print("batch:", tuple(x.shape), "label dist:", PCMMD_label_dist(train_csv))


def PCMMD_label_dist(csv):
    df = pd.read_csv(csv)
    return df["label"].value_counts().to_dict()


if __name__ == "__main__":
    import tempfile

    for fn in (test_splits_no_leakage_and_ratios, test_dataloader_batch_shape):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("OK: phase 2 smoke passed")
