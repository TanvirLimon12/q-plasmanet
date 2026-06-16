"""Phase 9 — assemble manuscript Tables (1-7) from committed result CSVs into
docs/tables.md. Dependency-free markdown (no tabulate). Run AFTER the result CSVs
are final.

  python -m src.viz.tables --config configs/base.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import add_common_args, load_config  # noqa: E402


def _md(df: pd.DataFrame, cols=None, round_to=4) -> str:
    if cols:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    df = df.round(round_to)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def build_tables(cfg) -> Path:
    repo = Path(cfg._repo_root)
    R = repo / cfg.get_path("paths.results")
    out = repo / "docs" / "tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    def rd(name):
        p = R / name
        return pd.read_csv(p) if p.is_file() else None

    parts = ["# Q-PlasmaNet — Manuscript Tables\n",
             "_Auto-generated from `results/*.csv`._\n"]

    specs = [
        ("Table 1 — Dataset summary (PCMMD)", "dataset_summary.csv",
         ["subset", "n_images", "n_plasma", "n_non_plasma", "n_patients", "most_common_wh"]),
        ("Table 2 — Quantum ablation", "quantum_ablation.csv",
         ["qubits", "depth", "entanglement", "encoding", "params", "f1_mean", "f1_std", "auc_mean"]),
        ("Table 3 — Classical baselines (5-seed)", "baselines.csv",
         ["model", "n_seeds", "trainable_params", "f1_mean", "f1_std", "auc_mean", "acc_mean", "ece_mean"]),
        ("Table 3b — Q-PlasmaNet vs matched heads (5-seed)", "quantum_main.csv",
         ["model", "params", "f1_mean", "f1_std", "auc_mean", "acc_mean", "ece_mean"]),
        ("Table 3c — Paired significance", "significance_main.csv",
         ["comparison", "mean_a", "mean_b", "mean_diff", "t_p", "wilcoxon_p"]),
        ("Table 5 — Robustness (retained-accuracy score)", "robustness_score.csv", None),
        ("Table 6 — NISQ shot/noise", "nisq.csv",
         ["study", "shots", "noise_type", "noise_p", "acc", "auc"]),
        ("Table 7 — Patient-level (per patient)", "patient_level.csv", None),
        ("Table 7b — Patient-level summary", "patient_level_summary.csv", None),
        ("Table 8 — External validation", "external_validation.csv",
         ["dataset", "head", "n_train", "f1", "auc", "acc"]),
        ("Table S — Few-shot", "fewshot.csv",
         ["k_per_class", "model", "f1_mean", "f1_std", "auc_mean"]),
        ("Table 9 — Effective dimension (Abbas 2021)", "effective_dim.csv",
         ["model", "n_params", "n_data", "eff_dim", "eff_dim_norm"]),
        ("Table 10 — Trainability (gradient variance)", "trainability.csv",
         ["n_qubits", "depth", "grad_var", "log10_var"]),
        ("Table 11 — Fourier expressivity (active freqs)", "expressivity.csv",
         ["layers", "frequency", "magnitude"]),
    ]
    for title, fname, cols in specs:
        df = rd(fname)
        parts.append(f"\n## {title}\n")
        parts.append(_md(df, cols) if df is not None else f"_(missing {fname} — run the owning phase)_")

    out.write_text("\n".join(parts) + "\n")
    print(f"[tables] wrote {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble manuscript tables (Phase 9).")
    add_common_args(p)
    args = p.parse_args()
    build_tables(load_config(args.config or ["configs/base.yaml"]))


if __name__ == "__main__":
    main()
