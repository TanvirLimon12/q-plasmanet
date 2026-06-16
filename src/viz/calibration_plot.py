"""Q1 figure — reliability diagram (confidence vs accuracy) + ECE for q_plasmanet
and the custom CNN on the PCMMD test split. Inference only.

Output: figures/fig_calibration.png, results/calibration.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import torch                     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import make_dataloader           # noqa: E402
from src.eval.calibration import compute_calibration   # noqa: E402
from src.models.qplasmanet import (                    # noqa: E402
    build_classical_infer, build_qplasmanet_infer)
from src.train.engine import predict                   # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402

# Blue palette (consistent with plots.py)
_B = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]
_MODEL_C = {"cnn": _B[1], "q_plasmanet": _B[0]}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "legend.framealpha": 0.85,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})


def run_calibration_plot(cfg, seed: int = 0) -> None:
    repo    = Path(cfg._repo_root)
    figures = repo / cfg.get_path("paths.figures"); figures.mkdir(parents=True, exist_ok=True)
    results = repo / cfg.get_path("paths.results")
    splits  = repo / cfg.get_path("paths.splits")
    dev     = torch.device("cpu")
    test_csv = splits / f"seed{seed}_test.csv"

    models = {
        "cnn":         build_classical_infer(cfg, "cnn", seed, device=dev),
        "q_plasmanet": build_qplasmanet_infer(cfg, seed, device=dev),
    }

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 6),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # Perfect calibration diagonal
    ax_rel.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect calibration")

    rows = []
    MIN_WEIGHT = 0.03   # skip bins containing <3% of test samples (too few for reliable acc)
    N_BINS = 7

    for name, model in models.items():
        dl  = make_dataloader(test_csv, cfg, train=False, batch_size=128, shuffle=False)
        y, p = predict(model, dl, dev)
        cal  = compute_calibration(y, p, n_bins=N_BINS)

        # Filter low-population bins and plot reliability curve
        valid_bins = [b for b in cal["bins"] if b["weight"] >= MIN_WEIGHT]
        xs = [b["conf"] for b in valid_bins]
        ys = [b["acc"]  for b in valid_bins]
        color = _MODEL_C.get(name, _B[2])
        label = f"{name.replace('_', ' ')} (ECE={cal['ece']:.3f})"
        ax_rel.plot(xs, ys, marker="o", color=color, linewidth=2,
                    markersize=7, label=label, zorder=3)

        # Confidence histogram on lower axes
        all_bins = cal["bins"]
        conf_c = [b["conf"] for b in all_bins]
        wts    = [b["weight"] for b in all_bins]
        bw = 1.0 / N_BINS * 0.35
        offset = -bw / 2 if name == "cnn" else bw / 2
        ax_hist.bar([c + offset for c in conf_c], wts, width=bw,
                    color=color, alpha=0.75, edgecolor="white")

        rows.append({"model": name, "ece": round(cal["ece"], 4),
                     "brier": round(cal["brier"], 4)})
        print(f"[calibration] {name}: ECE={cal['ece']:.4f} Brier={cal['brier']:.4f}",
              flush=True)

    ax_rel.set_ylabel("Fraction positive", fontsize=13)
    ax_rel.set_xlim(0, 1); ax_rel.set_ylim(0, 1)
    ax_rel.set_title("Reliability diagram (PCMMD test)", fontsize=14, fontweight="bold")
    ax_rel.legend(loc="upper left", fontsize=11)
    ax_rel.grid(True, alpha=0.18, linestyle=":")
    ax_rel.set_xticklabels([])   # shared with hist below
    ax_rel.tick_params(labelsize=12)
    ax_rel.text(0.02, 0.98, "(a)", transform=ax_rel.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")

    ax_hist.set_xlabel("Mean predicted confidence", fontsize=13)
    ax_hist.set_ylabel("Fraction\nof samples", fontsize=11)
    ax_hist.set_xlim(0, 1)
    ax_hist.tick_params(labelsize=12)
    ax_hist.text(0.02, 0.93, "(b)", transform=ax_hist.transAxes,
                 fontsize=14, fontweight="bold", va="top", ha="left")
    ax_hist.grid(True, alpha=0.18, linestyle=":")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.savefig(figures / "fig_calibration.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(figures / "fig_calibration.pdf", bbox_inches="tight",
                facecolor="white")   # vector for camera-ready
    plt.close(fig)
    pd.DataFrame(rows).to_csv(results / "calibration.csv", index=False)
    print("[calibration] wrote figures/fig_calibration.png + results/calibration.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Reliability diagram + ECE (Q1).")
    add_common_args(p)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml", "configs/quantum.yaml"])
    cfg["feature_backbone"] = "cnn"
    run_calibration_plot(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
