"""Q1 analysis — Fourier expressivity of the data-reuploading head (Schuld et al.,
Phys. Rev. A 103:032430, 2021).

A re-uploading model is a truncated Fourier series in each input coordinate; the
number of re-upload layers L sets the highest accessible frequency. We sweep one
input coordinate over [-π, π], read the circuit expectation, FFT it, and show the
magnitude spectrum grows richer with L — the mechanism behind the head reaching
classical-parity accuracy.

Output: results/expressivity.csv (layers, frequency, magnitude)
        figures/fig_expressivity.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.quantum_head import entangle_pairs  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402


def _circuit_fn(n, layers, entanglement="circular"):
    dev = qml.device("lightning.qubit", wires=n)
    ent = entangle_pairs(n, entanglement)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circuit(inputs, scaling, weights):
        for l in range(layers):
            f = scaling[l] * inputs
            for q in range(n):
                qml.Rot(f[..., 3 * q], f[..., 3 * q + 1], f[..., 3 * q + 2], wires=q)
            for q in range(n):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
            for a, b in ent:
                qml.CNOT(wires=[a, b])
        return qml.expval(qml.PauliZ(0))

    return circuit


def run_expressivity(cfg, n=8, layers_list=(1, 2, 3), grid=64, n_inits=8, seed=0) -> pd.DataFrame:
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results"); results.mkdir(parents=True, exist_ok=True)
    figures = repo / cfg.get_path("paths.figures"); figures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ts = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    in_dim = 3 * n

    rows = []
    spectra = {}
    for L in layers_list:
        circuit = _circuit_fn(n, L)
        mag = np.zeros(grid // 2)
        for _ in range(n_inits):
            scaling = torch.tensor(rng.normal(1.0, 0.1, (L, in_dim)), dtype=torch.float64)
            weights = torch.tensor(rng.uniform(-np.pi, np.pi, (L, n, 3)), dtype=torch.float64)
            x0 = torch.tensor(rng.uniform(-np.pi, np.pi, in_dim), dtype=torch.float64)
            ys = []
            for t in ts:
                x = x0.clone(); x[0] = t                 # sweep coordinate 0
                ys.append(float(circuit(x, scaling, weights)))
            coeffs = np.abs(np.fft.rfft(np.asarray(ys)))[: grid // 2]
            mag += coeffs / n_inits
        spectra[L] = mag
        for fq, m in enumerate(mag):
            rows.append({"layers": L, "frequency": fq, "magnitude": float(m)})
        nonzero = int((mag > 0.01 * mag.max()).sum())
        print(f"[expressivity] L={L}: ~{nonzero} active frequencies", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(results / "expressivity.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _B = ["#08306b", "#2171b5", "#4292c6", "#6baed6"]
        _LS = ["-", "--", "-."]
        _MK = ["o", "s", "^"]
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
            "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
            "figure.facecolor": "white", "savefig.facecolor": "white",
        })
        fig, ax = plt.subplots(figsize=(5, 4))
        zoom = 16   # only first 16 frequencies are informative
        for idx, (L, mag) in enumerate(spectra.items()):
            ax.plot(range(zoom), mag[:zoom],
                    marker=_MK[idx % len(_MK)], ls=_LS[idx % len(_LS)],
                    color=_B[idx % len(_B)], label=f"L={L} re-uploads",
                    linewidth=2, markersize=5)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Mean |Fourier coefficient|")
        ax.set_title("Re-uploading expands accessible Fourier spectrum")
        ax.set_yscale("log")
        ax.legend(); ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(figures / "fig_expressivity.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[expressivity] plot skipped: {e}")
    print(f"[expressivity] wrote {results/'expressivity.csv'}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Fourier expressivity vs re-upload layers (Schuld).")
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/quantum.yaml"])
    if args.smoke:
        run_expressivity(cfg, grid=32, n_inits=3)
    else:
        run_expressivity(cfg)


if __name__ == "__main__":
    main()
