"""Q1 analysis — accuracy vs training-set size: does the quantum head's higher
capacity ever pay off as data shrinks? Trains q-reupload / matched-MLP / logistic
on fractions of the PCMMD train split (trained-cnn d24 features), evals full test,
R draws per fraction. Maps the (non-)crossover that explains the parity.

Output: results/data_scaling.csv, figures/fig_data_scaling.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import compute_metrics  # noqa: E402
from src.features.precompute import load_features  # noqa: E402
from src.models.classical_heads import mlp_for_param_budget  # noqa: E402
from src.models.quantum_head import build_reupload_head  # noqa: E402
from src.train.engine import predict, train_loop  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402


def _fit_eval(model, Xk, yk, Xte, yte, epochs=60):
    dl = DataLoader(TensorDataset(torch.tensor(Xk, dtype=torch.float32),
                                  torch.tensor(yk, dtype=torch.long)),
                    batch_size=min(64, len(Xk)), shuffle=True)
    best, _ = train_loop(model, dl, None, epochs=epochs, lr=0.03, seed=0,
                         device=torch.device("cpu"), verbose=False)
    model.load_state_dict(best)
    te = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                                  torch.tensor(yte, dtype=torch.long)), batch_size=256)
    yt, pt = predict(model, te, torch.device("cpu"))
    return compute_metrics(yt, pt)


def run_data_scaling(cfg, fractions=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0), draws=3, seed=0):
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results"); results.mkdir(parents=True, exist_ok=True)
    figures = repo / cfg.get_path("paths.figures"); figures.mkdir(parents=True, exist_ok=True)
    bb = cfg.get_path("feature_backbone") or "cnn"
    Xtr, ytr = load_features(cfg, bb, 24, seed, "train")
    Xte, yte = load_features(cfg, bb, 24, seed, "test")
    if Xtr is None:
        raise FileNotFoundError("need d24 features (precompute --from-classifier cnn).")
    ytr = ytr.astype(int); yte = yte.astype(int)
    i0, i1 = np.where(ytr == 0)[0], np.where(ytr == 1)[0]

    rows = []
    for frac in fractions:
        k = max(2, int(len(ytr) * frac / 2))             # per class
        acc = {m: [] for m in ("logistic", "mlp_param_matched", "q_plasmanet")}
        for d in range(draws):
            rng = np.random.default_rng(d)
            sel = np.concatenate([rng.choice(i0, min(k, len(i0)), replace=False),
                                  rng.choice(i1, min(k, len(i1)), replace=False)])
            Xk, yk = Xtr[sel], ytr[sel]
            lr = LogisticRegression(max_iter=2000).fit(Xk, yk)
            acc["logistic"].append(compute_metrics(yte, lr.predict_proba(Xte)[:, 1])["f1"])
            acc["mlp_param_matched"].append(_fit_eval(mlp_for_param_budget(24, 178, 2), Xk, yk, Xte, yte)["f1"])
            q, _ = build_reupload_head(8, 3, "circular", "lightning.qubit", "adjoint", verify=False)
            acc["q_plasmanet"].append(_fit_eval(q, Xk, yk, Xte, yte)["f1"])
        for m, v in acc.items():
            rows.append({"fraction": frac, "n_per_class": k, "model": m,
                         "f1_mean": float(np.mean(v)), "f1_std": float(np.std(v))})
        print(f"[data_scaling] frac={frac} (k={k}/cls): " +
              " ".join(f"{m}={np.mean(v):.3f}" for m, v in acc.items()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(results / "data_scaling.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _B = ["#08306b", "#2171b5", "#4292c6", "#6baed6"]
        _MC = {"logistic": _B[3], "mlp_param_matched": _B[2], "q_plasmanet": _B[0]}
        _LS = ["-", "--", "-."]
        _MK = ["o", "s", "^"]
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
            "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
            "figure.facecolor": "white", "savefig.facecolor": "white",
        })
        fig, ax = plt.subplots(figsize=(5, 4))
        for idx, m in enumerate(sorted(df.model.unique())):
            s = df[df.model == m].sort_values("fraction")
            ax.errorbar(s["fraction"], s["f1_mean"], yerr=s["f1_std"],
                        marker=_MK[idx % len(_MK)], ls=_LS[idx % len(_LS)],
                        capsize=4, label=m.replace("_", " "),
                        color=_MC.get(m, _B[idx % len(_B)]), linewidth=2, markersize=6)
        ax.set_xscale("log")
        ax.set_xlabel("Train fraction")
        ax.set_ylabel("Test F1")
        ax.set_title("Accuracy vs training-set size")
        ax.legend(); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25, linestyle="--")
        fig.tight_layout()
        fig.savefig(figures / "fig_data_scaling.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[data_scaling] plot skipped: {e}")
    print(f"[data_scaling] wrote results/data_scaling.csv")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Accuracy vs train size (Q1 crossover).")
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml", "configs/quantum.yaml"])
    cfg["feature_backbone"] = "cnn"
    run_data_scaling(cfg)


if __name__ == "__main__":
    main()
