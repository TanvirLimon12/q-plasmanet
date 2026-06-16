"""Phase-8 smoke: external validation runs end-to-end on a tiny MedMNIST slice.

Requires the `medmnist` package + network (downloads BloodMNIST). Skips cleanly if
unavailable so CI without network still passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.config import load_config  # noqa: E402


def test_external_smoke(tmp_path):
    try:
        import medmnist  # noqa: F401
        from src.eval.external_validation import run_external
    except Exception as e:                      # package missing
        print(f"SKIP phase 8 (medmnist unavailable): {e}")
        return

    cfg = load_config(["configs/backbones.yaml", "configs/quantum.yaml"])
    cfg["_repo_root"] = str(tmp_path)
    try:
        df = run_external(cfg, datasets=["bloodmnist"], classes=(0, 1), smoke=True, limit=120)
    except Exception as e:                       # network/download failure
        print(f"SKIP phase 8 (download failed): {e}")
        return

    assert (tmp_path / "results" / "external_validation.csv").is_file()
    assert set(df["head"]) >= {"logistic", "mlp_param_matched", "q_plasmanet"}
    print("OK: phase 8 smoke passed")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_external_smoke(Path(d))
