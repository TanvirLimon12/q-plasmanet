"""Phase 9 — figures generated from committed result CSVs.

Produces (when the underlying CSV exists):
  figures/fig_model_comparison.png  — F1 ± std: CNN vs MLP vs Q-PlasmaNet
  figures/fig_robustness.png        — accuracy vs corruption severity
  figures/fig_nisq_shots.png        — accuracy/AUC vs shots
  figures/fig_nisq_noise.png        — accuracy under each hardware-noise channel
  figures/fig_fewshot.png           — F1 vs labels/class (few-shot study)
  figures/fig_external.png          — external validation (BloodMNIST/PathMNIST)
  figures/fig_effective_dim.png     — normalized effective dimension
  figures/fig_ablation.png          — quantum ablation heatmap (qubits × depth)
  figures/fig_separability.png      — feature separability: linear probe AUC + PR
  figures/fig_patient.png           — patient-level scatter (predicted vs true ratio)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import add_common_args, load_config  # noqa: E402

# Journal-quality global style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.framealpha": 0.85,
    "lines.linewidth": 2.2,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# Blue palette (sequential, dark → light)
_B = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]

# Consistent model colours across all figures
_MODEL_C: dict[str, str] = {
    "cnn":               _B[1],
    "CNN":               _B[1],
    "resnet18":          _B[2],
    "mobilenet_v2":      _B[3],
    "efficientnet_b0":   _B[4],
    "mlp_param_matched": _B[2],
    "MLP (matched)":     _B[2],
    "mlp_matched":       _B[2],
    "q_plasmanet":       _B[0],
    "Q-PlasmaNet":       _B[0],
    "q_reupload":        _B[0],
    "logistic":          _B[3],
    "Logistic":          _B[3],
}

# Pretty, publication-quality legend/axis labels for model keys.
_LABELS = {
    "cnn": "CNN", "logistic": "Logistic", "mlp_matched": "Matched MLP",
    "mlp_param_matched": "Matched MLP", "q_plasmanet": "Q-PlasmaNet",
    "q_reupload": "Q-PlasmaNet", "mobilenet_v2": "MobileNetV2",
    "resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0",
}


def _pretty(name: str) -> str:
    return _LABELS.get(name, name.replace("_", " "))

# Corruption line colours (all blue-family, distinct linestyles added in code)
_CORR_C: dict[str, str] = {
    "blur":           _B[1],
    "brightness":     _B[2],
    "contrast":       _B[3],
    "gaussian_noise": _B[4],
    "jpeg":           _B[5],
    "stain":          _B[0],  # darkest — worst corruption
}

_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
_MARKERS    = ["o", "s", "^", "D", "v", "P"]


def _read(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.is_file() else None


def _mc(name: str) -> str:
    return _MODEL_C.get(name, _B[2])


def _save(fig, path: Path, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")   # vector for camera-ready
    plt.close(fig)


# Existing figures (rewritten with blue palette + fixes)

def plot_model_comparison(results: Path, figures: Path) -> None:
    bl = _read(results / "baselines.csv")
    qm = _read(results / "quantum_main.csv")
    if bl is None and qm is None:
        return
    names, f1_vals, err_vals = [], [], []
    if bl is not None and "cnn" in set(bl["model"]):
        r = bl[bl.model == "cnn"].iloc[0]
        names.append("CNN"); f1_vals.append(r["f1_mean"]); err_vals.append(r.get("f1_std", 0))
    if qm is not None:
        for m, lab in [("mlp_param_matched", "MLP (matched)"), ("q_plasmanet", "Q-PlasmaNet")]:
            if m in set(qm["model"]):
                r = qm[qm.model == m].iloc[0]
                names.append(lab); f1_vals.append(r["f1_mean"]); err_vals.append(r.get("f1_std", 0))
    if not names:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = [_mc(n) for n in names]
    ax.bar(names, f1_vals, yerr=err_vals, capsize=6, color=colors,
           edgecolor="white", linewidth=0.8,
           error_kw={"ecolor": "#333", "elinewidth": 1.5})
    f_min = max(0.0, min(v - e for v, e in zip(f1_vals, err_vals)) - 0.02)
    ax.set_ylabel("Test F1 (mean ± std, 5 seeds)")
    ax.set_ylim(f_min, 1.01)
    ax.set_title("Model comparison on PCMMD")
    for i, (v, e) in enumerate(zip(f1_vals, err_vals)):
        ax.text(i, v + e + 0.003, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    _save(fig, figures / "fig_model_comparison.png")


def plot_robustness(results: Path, figures: Path) -> None:
    df = _read(results / "robustness.csv")
    if df is None:
        return
    corr = df[df.corruption != "clean"]
    models = sorted(corr["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 4.5),
                             squeeze=False, sharey=True)
    for ax, model in zip(axes[0], models):
        sub = corr[corr.model == model]
        for idx, c in enumerate(sorted(sub["corruption"].unique())):
            s = sub[sub.corruption == c].sort_values("severity")
            ax.plot(s["severity"], s["acc_mean"],
                    marker=_MARKERS[idx % len(_MARKERS)],
                    ls=_LINESTYLES[idx % len(_LINESTYLES)],
                    color=_CORR_C.get(c, _B[idx % len(_B)]),
                    label=c.replace("_", " ").capitalize(), linewidth=2, markersize=6)
        _name_map = {"cnn": "CNN", "q_plasmanet": "Q-PlasmaNet",
                     "resnet18": "ResNet-18", "efficientnet_b0": "EfficientNet-B0",
                     "mobilenet_v2": "MobileNet-V2"}
        ax.set_title(_name_map.get(model, model.replace("_", " ")))
        ax.set_xlabel("Corruption severity")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=9, loc="lower left")
    axes[0][0].set_ylabel("Accuracy")
    fig.suptitle("Robustness: accuracy vs corruption severity", fontweight="bold")
    _save(fig, figures / "fig_robustness.png")


def plot_nisq(results: Path, figures: Path) -> None:
    df = _read(results / "nisq.csv")
    if df is None:
        return
    shots_df = df[df.study == "shots"].sort_values("shots")
    if not shots_df.empty:
        g = shots_df.groupby("shots")[["acc", "auc"]].mean().reset_index()
        lo = max(0.0, min(g["acc"].min(), g["auc"].min()) - 0.03)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(g["shots"], g["acc"], marker="o", color=_B[0],
                label="Accuracy", linewidth=2, markersize=7)
        ax.plot(g["shots"], g["auc"], marker="s", color=_B[2],
                label="AUC", linewidth=2, markersize=7, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("Number of shots")
        ax.set_ylabel("Score")
        ax.set_ylim(lo, 1.005)
        ax.set_title("NISQ shot-noise sensitivity")
        ax.legend()
        _save(fig, figures / "fig_nisq_shots.png")

    noise_df = df[df.study == "noise"]
    if not noise_df.empty:
        g = noise_df.groupby("noise_type")["acc"].mean().sort_values()
        lo = max(0.0, g.values.min() - 0.06)
        _chan = {"amplitude_damping": "Amplitude\ndamping", "phase_damping": "Phase\ndamping",
                 "depolarizing": "Depolarizing", "readout_error": "Readout\nerror"}
        labels = [_chan.get(n, n.replace("_", " ").capitalize()) for n in g.index]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(range(len(g)), g.values,
                      color=[_B[i % len(_B)] for i in range(len(g))],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, g.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(lo, 1.0)
        ax.set_title("Accuracy under hardware-noise channels")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(labels)
        _save(fig, figures / "fig_nisq_noise.png")


def plot_fewshot(results: Path, figures: Path) -> None:
    df = _read(results / "fewshot.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, m in enumerate(sorted(df["model"].unique())):
        s = df[df.model == m].sort_values("k_per_class")
        ax.errorbar(s["k_per_class"], s["f1_mean"], yerr=s["f1_std"],
                    marker=_MARKERS[idx % len(_MARKERS)],
                    ls=_LINESTYLES[idx % len(_LINESTYLES)],
                    capsize=4, label=_pretty(m),
                    color=_mc(m), linewidth=2, markersize=6)
    ax.set_xlabel("Labels per class")
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Few-shot sample efficiency")
    ax.legend()
    _save(fig, figures / "fig_fewshot.png")


def plot_external(results: Path, figures: Path) -> None:
    df = _read(results / "external_validation.csv")
    if df is None:
        return
    datasets = sorted(df["dataset"].unique())
    heads = ["logistic", "mlp_param_matched", "q_plasmanet"]
    head_labels = {
        "logistic": "Logistic",
        "mlp_param_matched": "MLP (matched)",
        "q_plasmanet": "Q-PlasmaNet",
    }
    x = np.arange(len(datasets))
    w = 0.22
    n_h = len(heads)
    offsets = np.linspace(-(n_h - 1) / 2 * w, (n_h - 1) / 2 * w, n_h)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, h in enumerate(heads):
        vals = [df[(df["dataset"] == d) & (df["head"] == h)]["f1"].mean()
                for d in datasets]
        bars = ax.bar(x + offsets[i], vals, w,
                      label=head_labels[h],
                      color=_mc(h), edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                        f"{v:.3f}", ha="center", fontsize=8)
    lo = max(0.0, min(
        df[(df["dataset"] == d) & (df["head"] == h)]["f1"].mean()
        for d in datasets for h in heads
        if not df[(df["dataset"] == d) & (df["head"] == h)].empty
    ) - 0.04)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("mnist", "MNIST").replace("blood", "Blood").replace("path", "Path")
         for d in datasets])
    ax.set_ylabel("Test F1")
    ax.set_ylim(lo, 1.02)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_title("External validation: head parity across datasets")
    ax.legend()
    _save(fig, figures / "fig_external.png")


def plot_effective_dim(results: Path, figures: Path) -> None:
    df = _read(results / "effective_dim.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in sorted(df["model"].unique()):
        s = df[df.model == m].sort_values("n_data")
        ax.plot(s["n_data"], s["eff_dim_norm"], marker="o",
                label=_pretty(m), color=_mc(m), linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("n (training samples)")
    ax.set_ylabel("Normalized effective dimension")
    ax.set_ylim(0, 1)
    ax.set_title("Effective dimension: quantum vs classical")
    ax.legend()
    _save(fig, figures / "fig_effective_dim.png")


# New journal figures

def plot_ablation(results: Path, figures: Path) -> None:
    """Quantum ablation heatmap: F1 vs qubits × depth, one panel per entanglement."""
    df = _read(results / "quantum_ablation.csv")
    if df is None:
        return
    angle = df[df["encoding"] == "angle"] if "encoding" in df.columns else df
    entanglements = sorted(angle["entanglement"].unique()) if "entanglement" in angle.columns else []
    if not entanglements:
        return

    vmin = float(angle["f1_mean"].min()) - 0.005
    vmax = float(angle["f1_mean"].max()) + 0.005
    mid  = (vmin + vmax) / 2

    fig, axes = plt.subplots(1, len(entanglements),
                             figsize=(4.2 * len(entanglements), 3.8),
                             squeeze=False)
    im = None
    for ax, ent in zip(axes[0], entanglements):
        sub = angle[angle["entanglement"] == ent]
        pivot = sub.pivot_table(index="depth", columns="qubits",
                                values="f1_mean").sort_index(ascending=False)
        im = ax.imshow(pivot.values, cmap="Blues",
                       vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Qubits")
        ax.set_ylabel("Depth")
        ax.set_title(f"{ent.capitalize()} entanglement")
        ax.grid(False)
        for r in range(len(pivot.index)):
            for c_idx in range(len(pivot.columns)):
                v = float(pivot.values[r, c_idx])
                txt_color = "white" if v > mid else "#08306b"
                ax.text(c_idx, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=txt_color)

    if im is not None:
        plt.colorbar(im, ax=axes[0][-1], label="Test F1", shrink=0.82, pad=0.02)
    fig.suptitle("Quantum ablation: F1 vs architecture (angle encoding)", fontweight="bold")
    _save(fig, figures / "fig_ablation.png")


def plot_separability(results: Path, figures: Path) -> None:
    """Feature separability: linear probe AUC and participation ratio per backbone."""
    df = _read(results / "separability.csv")
    if df is None:
        return
    backbones = list(df["backbone"].unique())
    x = np.arange(len(backbones))
    w = 0.3
    fig, ax1 = plt.subplots(figsize=(5.5, 4))
    ax2 = ax1.twinx()

    auc_vals = [float(df[df.backbone == b]["linear_probe_auc"].iloc[0]) for b in backbones]
    pr_vals  = [float(df[df.backbone == b]["participation_ratio"].iloc[0]) for b in backbones]

    b1 = ax1.bar(x - w / 2, auc_vals, w, label="Linear probe AUC",
                 color=_B[0], edgecolor="white")
    b2 = ax2.bar(x + w / 2, pr_vals,  w, label="Participation ratio",
                 color=_B[3], edgecolor="white")

    for bar, v in zip(b1, auc_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                 f"{v:.3f}", ha="center", fontsize=9, color=_B[0], fontweight="bold")
    for bar, v in zip(b2, pr_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                 f"{v:.1f}", ha="center", fontsize=9, color=_B[3], fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([b.replace("_", "-") for b in backbones])
    ax1.set_ylabel("Linear probe AUC", color=_B[0])
    ax2.set_ylabel("Participation ratio (intrinsic dim.)", color=_B[3])
    ax1.tick_params(axis="y", labelcolor=_B[0])
    ax2.tick_params(axis="y", labelcolor=_B[3])
    ax1.set_ylim(0.8, 1.05)
    ax1.set_title("Feature separability by backbone")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    _save(fig, figures / "fig_separability.png")


def plot_patient(results: Path, figures: Path) -> None:
    """Patient-level scatter: predicted vs true plasma ratio, coloured by diagnosis."""
    df = _read(results / "patient_level.csv")
    if df is None:
        return
    has_diag = "diagnosis" in df.columns
    fig, ax = plt.subplots(figsize=(5, 5))

    if has_diag:
        diag_map = [("diseased", _B[0], "Diseased"), ("normal", _B[3], "Normal")]
        for diag, color, label in diag_map:
            sub = df[df["diagnosis"] == diag]
            if sub.empty:
                continue
            ax.scatter(sub["true_plasma_ratio"], sub["pred_plasma_ratio"],
                       c=color, s=95, label=label,
                       edgecolors="white", linewidths=0.9, zorder=3)
            for _, row in sub.iterrows():
                # offset label away from the cluster centre to reduce overlap
                dx = 7 if row["true_plasma_ratio"] >= 0.2 else -7
                ha = "left" if dx > 0 else "right"
                ax.annotate(f"P{int(row['patient'])}",
                            (row["true_plasma_ratio"], row["pred_plasma_ratio"]),
                            textcoords="offset points", xytext=(dx, 4),
                            fontsize=7.5, ha=ha, color=color)
    else:
        ax.scatter(df["true_plasma_ratio"], df["pred_plasma_ratio"],
                   c=_B[1], s=95, edgecolors="white", linewidths=0.9, zorder=3)

    hi = max(df["true_plasma_ratio"].max(), df["pred_plasma_ratio"].max()) * 1.12
    ax.plot([0, hi], [0, hi], ls="--", color="#888", linewidth=1.3,
            alpha=0.7, label="Perfect prediction", zorder=1)
    r = float(np.corrcoef(df["true_plasma_ratio"], df["pred_plasma_ratio"])[0, 1])
    ax.text(0.57, 0.08, f"Pearson r = {r:.3f}",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color=_B[0])
    ax.set_xlabel("True plasma cell ratio")
    ax.set_ylabel("Predicted plasma cell ratio")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_title("Patient-level plasma ratio: predicted vs true")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal")
    _save(fig, figures / "fig_patient.png")


# Analysis figures regenerated from existing CSVs

def plot_trainability(results: Path, figures: Path) -> None:
    """Barren-plateau: gradient variance vs qubits, one line per depth."""
    df = _read(results / "trainability.csv")
    if df is None:
        return
    depths  = sorted(df["depth"].unique())
    all_q   = sorted(df["n_qubits"].unique())
    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, d in enumerate(depths):
        s = df[df.depth == d].sort_values("n_qubits")
        ax.semilogy(s["n_qubits"], s["grad_var"],
                    marker=_MARKERS[idx % len(_MARKERS)],
                    ls=_LINESTYLES[idx % len(_LINESTYLES)],
                    color=_B[idx % len(_B)], label=f"Depth {d}",
                    linewidth=2, markersize=7)
    ax.set_xticks(all_q)
    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Var[∂C/∂θ]  (log scale)")
    ax.set_title("Trainability: gradient variance vs qubits")
    ax.legend()
    _save(fig, figures / "fig_trainability.png")


def plot_expressivity(results: Path, figures: Path) -> None:
    """Fourier expressivity: mean |coeff| vs frequency per re-upload depth L."""
    df = _read(results / "expressivity.csv")
    if df is None:
        return
    layers_list = sorted(df["layers"].unique())
    zoom = 16   # first 16 frequencies are informative; rest are negligible
    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, L in enumerate(layers_list):
        sub = df[df.layers == L].sort_values("frequency")
        sub = sub[sub.frequency < zoom]
        ax.plot(sub["frequency"], sub["magnitude"],
                marker=_MARKERS[idx % len(_MARKERS)],
                ls=_LINESTYLES[idx % len(_LINESTYLES)],
                color=_B[idx % len(_B)], label=f"L={L} re-uploads",
                linewidth=2, markersize=5)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Mean |Fourier coefficient|")
    ax.set_title("Re-uploading expands accessible Fourier spectrum")
    ax.legend()
    _save(fig, figures / "fig_expressivity.png")


def plot_data_scaling(results: Path, figures: Path) -> None:
    """Accuracy vs training-set size across heads."""
    df = _read(results / "data_scaling.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    for idx, m in enumerate(sorted(df.model.unique())):
        s = df[df.model == m].sort_values("fraction")
        ax.errorbar(s["fraction"], s["f1_mean"], yerr=s["f1_std"],
                    marker=_MARKERS[idx % len(_MARKERS)],
                    ls=_LINESTYLES[idx % len(_LINESTYLES)],
                    capsize=4, label=_pretty(m),
                    color=_mc(m), linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Test F1")
    ax.set_title("Accuracy vs training-set size")
    ax.legend()
    ax.set_ylim(0, 1)
    _save(fig, figures / "fig_data_scaling.png")


# Orchestration

def make_plots(cfg) -> list[str]:
    repo    = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results")
    figures = repo / cfg.get_path("paths.figures")
    figures.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(results, figures)
    plot_robustness(results, figures)
    plot_nisq(results, figures)
    plot_fewshot(results, figures)
    plot_external(results, figures)
    plot_effective_dim(results, figures)
    plot_ablation(results, figures)
    plot_separability(results, figures)
    plot_patient(results, figures)
    plot_trainability(results, figures)
    plot_expressivity(results, figures)
    plot_data_scaling(results, figures)

    made = sorted(p.name for p in figures.glob("*.png"))
    print(f"[plots] wrote {len(made)} figures to {figures}: {made}")
    return made


def main() -> None:
    p = argparse.ArgumentParser(description="Generate figures from result CSVs (Phase 9).")
    add_common_args(p)
    args = p.parse_args()
    make_plots(load_config(args.config or ["configs/base.yaml"]))


if __name__ == "__main__":
    main()
