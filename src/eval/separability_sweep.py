"""Separability sweep — turns the parity null result into a predictive criterion.

Claim under test: a higher-capacity head (the matched MLP or the re-uploading
quantum head) can only beat a linear probe when the frozen features are NOT
already linearly separable, and the size of that gain is predicted by the
linear-probe AUC of the task.

We hold the real PCMMD trained-cnn d24 features fixed and vary the *separability
of the task* with a single knob: a frequency omega. A random unit direction w is
fixed; the projection t = w . x is scaled to [-pi, pi]; the binary label is
y = 1[ sin(omega * t) > 0 ]. Small omega -> a near-linear boundary (linearly
separable, high linear-probe AUC); large omega -> a high-frequency boundary that
a linear probe cannot fit (low linear-probe AUC) but an expressive head can.
This directly connects to the Fourier-expressivity view of re-uploading circuits.

At each omega we train logistic / matched-MLP / q-reupload on the SAME features
and report test F1, plus the logistic (linear-probe) AUC of the task. The gap of
each capacity head over logistic, plotted against linear-probe AUC, is the
criterion curve.

Output: results/separability_sweep.csv, figures/fig_sep_sweep.png

  python -m src.eval.separability_sweep --config configs/backbones.yaml configs/quantum.yaml
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


def _make_task(Xtr, Xte, omega, rng):
    """Binary label from a sum of per-coordinate sinusoids with random signs,
    y = 1[ sum_j a_j sin(omega * s(x_j)) > median ]. This is the native
    hypothesis class of a re-uploading VQC (Fourier series per input coordinate),
    so the knob is fair to both the quantum head and the MLP. Low omega -> the
    summand is ~linear in each coordinate (linearly separable, high linear-probe
    AUC); high omega -> high-frequency per-coordinate structure that a linear
    probe cannot fit. Per-coordinate scaling fit on train only; threshold (the
    train median) keeps the task balanced."""
    d = Xtr.shape[1]
    a = rng.choice([-1.0, 1.0], size=d)
    lo, hi = Xtr.min(0), Xtr.max(0)
    s = lambda X: np.pi * (2 * (np.clip(X, lo, hi) - lo) / (hi - lo + 1e-9) - 1)
    gtr = (np.sin(omega * s(Xtr)) * a).sum(1)
    gte = (np.sin(omega * s(Xte)) * a).sum(1)
    thr = np.median(gtr)
    return (gtr > thr).astype(int), (gte > thr).astype(int)


def run_sep_sweep(cfg, omegas=(0.5, 1.0, 2.0, 3.0, 4.0, 6.0), draws=2,
                  seed=0, backbone="cnn", n_cap=600, epochs=30):
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results"); results.mkdir(parents=True, exist_ok=True)
    figures = repo / cfg.get_path("paths.figures"); figures.mkdir(parents=True, exist_ok=True)
    Xtr, _ = load_features(cfg, backbone, 24, seed, "train")
    Xte, _ = load_features(cfg, backbone, 24, seed, "test")
    if Xtr is None or Xte is None:
        raise FileNotFoundError(f"need {backbone} d24 features (precompute --from-classifier cnn).")
    # standardize features on train (so the projection scale is stable)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    rows = []
    for om in omegas:
        acc = {m: {"f1": [], "auc": []} for m in ("logistic", "mlp_matched", "q_reupload")}
        for d in range(draws):
            rng = np.random.default_rng(1000 + d)
            ytr, yte = _make_task(Xtr, Xte, om, rng)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            # cap the quantum/MLP training set (balanced) to keep wall-clock tractable;
            # logistic and evaluation use the full splits.
            i0, i1 = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
            k = min(n_cap // 2, len(i0), len(i1))
            sel = np.concatenate([rng.choice(i0, k, replace=False),
                                  rng.choice(i1, k, replace=False)])
            Xc, yc = Xtr[sel], ytr[sel]
            lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            ml = compute_metrics(yte, lr.predict_proba(Xte)[:, 1])
            acc["logistic"]["f1"].append(ml["f1"]); acc["logistic"]["auc"].append(ml["auc"])
            mm = _fit_eval(mlp_for_param_budget(24, 178, 2), Xc, yc, Xte, yte, epochs=epochs)
            acc["mlp_matched"]["f1"].append(mm["f1"]); acc["mlp_matched"]["auc"].append(mm["auc"])
            q, _ = build_reupload_head(8, 3, "circular", "lightning.qubit", "adjoint", verify=False)
            qm = _fit_eval(q, Xc, yc, Xte, yte, epochs=epochs)
            acc["q_reupload"]["f1"].append(qm["f1"]); acc["q_reupload"]["auc"].append(qm["auc"])
        lp_auc = float(np.mean(acc["logistic"]["auc"]))   # linear-probe AUC of the task
        row = {"omega": om, "linear_probe_auc": lp_auc}
        for m in acc:
            row[f"{m}_f1"] = float(np.mean(acc[m]["f1"]))
            row[f"{m}_f1_std"] = float(np.std(acc[m]["f1"]))
        row["gap_mlp_vs_lin"] = row["mlp_matched_f1"] - row["logistic_f1"]
        row["gap_q_vs_lin"] = row["q_reupload_f1"] - row["logistic_f1"]
        row["gap_q_vs_mlp"] = row["q_reupload_f1"] - row["mlp_matched_f1"]
        rows.append(row)
        print(f"[sep_sweep] omega={om:>4}: lin-probe AUC={lp_auc:.3f} | "
              f"log={row['logistic_f1']:.3f} mlp={row['mlp_matched_f1']:.3f} "
              f"q={row['q_reupload_f1']:.3f} | gap_q-lin={row['gap_q_vs_lin']:+.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(results / "separability_sweep.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _B = ["#08306b", "#2171b5", "#6baed6"]
        plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
                             "axes.titleweight": "bold", "axes.spines.top": False,
                             "axes.spines.right": False, "figure.facecolor": "white"})
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.3))
        # (a) F1 vs linear-probe AUC
        s = df.sort_values("linear_probe_auc")
        axA.plot(s.linear_probe_auc, s.logistic_f1, "o-", color=_B[2], lw=2.4, label="Logistic (linear)")
        axA.plot(s.linear_probe_auc, s.mlp_matched_f1, "s--", color=_B[1], lw=2.4, label="Matched MLP")
        axA.plot(s.linear_probe_auc, s.q_reupload_f1, "^-", color=_B[0], lw=2.4, label="Q-PlasmaNet")
        axA.set_xlabel("Linear-probe AUC of task (separability)")
        axA.set_ylabel("Test F1"); axA.set_title("(a) Capacity helps as separability drops")
        axA.invert_xaxis(); axA.grid(True, alpha=0.18, ls=":"); axA.legend(fontsize=11)
        axA.text(0.02, 0.02, "(a)", transform=axA.transAxes, fontsize=14, fontweight="bold")
        # (b) gap over logistic vs linear-probe AUC
        axB.axhline(0, color="gray", lw=1, ls="--")
        axB.plot(s.linear_probe_auc, s.gap_q_vs_lin, "^-", color=_B[0], lw=2.4, label="Q $-$ logistic")
        axB.plot(s.linear_probe_auc, s.gap_mlp_vs_lin, "s--", color=_B[1], lw=2.4, label="MLP $-$ logistic")
        axB.set_xlabel("Linear-probe AUC of task")
        axB.set_ylabel(r"$\Delta$F1 (head $-$ logistic)")
        axB.set_title("(b) Gain predicted by linear-probe AUC")
        axB.invert_xaxis(); axB.grid(True, alpha=0.18, ls=":"); axB.legend(fontsize=11)
        fig.subplots_adjust(wspace=0.28)
        axB.text(0.02, 0.02, "(b)", transform=axB.transAxes, fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(figures / "fig_sep_sweep.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures / "fig_sep_sweep.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[sep_sweep] plot skipped: {e}")
    print("[sep_sweep] wrote results/separability_sweep.csv")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Separability sweep: capacity helps iff not linearly separable.")
    add_common_args(p)
    p.add_argument("--draws", type=int, default=3)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml", "configs/quantum.yaml"])
    cfg["feature_backbone"] = "cnn"
    run_sep_sweep(cfg, draws=args.draws)


if __name__ == "__main__":
    main()
