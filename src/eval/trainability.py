"""Q1 analysis — trainability / barren-plateau (gradient variance vs qubits/depth).

Protocol (McClean et al., Nat. Commun. 2018): for each (n_qubits, depth), draw R
random parameter sets + random inputs, evaluate the gradient of a fixed cost
(<Z_0>) wrt one ansatz parameter, and report Var over the R samples. A variance
that decays exponentially in n_qubits indicates a barren plateau. We use the same
RX/RY/RZ+CNOT ansatz as the angle head, isolated from the linear readout.

Output: results/trainability.csv  (n_qubits, depth, grad_var, grad_mean, log10_var)
        figures/fig_trainability.png
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


def _grad_sample(n, depth, entanglement, rng, device_name="lightning.qubit"):
    dev = qml.device(device_name, wires=n)
    pairs = entangle_pairs(n, entanglement)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
        for layer in range(depth):
            for w in range(n):
                qml.RX(weights[layer, w, 0], wires=w)
                qml.RY(weights[layer, w, 1], wires=w)
                qml.RZ(weights[layer, w, 2], wires=w)
            for a, b in pairs:
                qml.CNOT(wires=[a, b])
        return qml.expval(qml.PauliZ(0))

    x = torch.tensor(rng.uniform(-np.pi, np.pi, n), dtype=torch.float64)
    w = torch.tensor(rng.uniform(-np.pi, np.pi, (depth, n, 3)), dtype=torch.float64,
                     requires_grad=True)
    out = circuit(x, w)
    out.backward()
    # gradient wrt a non-degenerate parameter: RY (k=1) of a middle qubit in the
    # LAST layer. The first RX on qubit 0 is symmetric under the RY embedding and
    # gives a spurious ~0 gradient at depth 1.
    return float(w.grad[depth - 1, n // 2, 1])


def run_trainability(cfg, qubits=(2, 4, 6, 8), depths=(1, 2, 4), entanglement="circular",
                     n_samples=100, seed=0) -> pd.DataFrame:
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results"); results.mkdir(parents=True, exist_ok=True)
    figures = repo / cfg.get_path("paths.figures"); figures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    for d in depths:
        for n in qubits:
            grads = [_grad_sample(n, d, entanglement, rng) for _ in range(n_samples)]
            v = float(np.var(grads))
            rows.append({"n_qubits": n, "depth": d, "grad_var": v,
                         "grad_mean": float(np.mean(grads)),
                         "log10_var": float(np.log10(v + 1e-300))})
            print(f"[trainability] n={n} depth={d} Var[grad]={v:.3e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(results / "trainability.csv", index=False)

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
        for idx, d in enumerate(depths):
            s = df[df.depth == d].sort_values("n_qubits")
            ax.semilogy(s["n_qubits"], s["grad_var"],
                        marker=_MK[idx % len(_MK)], ls=_LS[idx % len(_LS)],
                        color=_B[idx % len(_B)], label=f"Depth {d}",
                        linewidth=2, markersize=7)
        all_q = sorted(df["n_qubits"].unique())
        ax.set_xticks(all_q)
        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("Var[∂C/∂θ] (log scale)")
        ax.set_title("Trainability: gradient variance vs qubits")
        ax.legend(); ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(figures / "fig_trainability.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[trainability] plot skipped: {e}")

    print(f"[trainability] wrote {results/'trainability.csv'}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Barren-plateau / gradient-variance analysis.")
    add_common_args(p)
    p.add_argument("--n-samples", type=int, default=100)
    args = p.parse_args()
    n = 20 if args.smoke else args.n_samples
    run_trainability(load_config(args.config or ["configs/quantum.yaml"]), n_samples=n)


if __name__ == "__main__":
    main()
