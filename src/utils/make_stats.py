"""Q1 analysis — consolidate bootstrap CIs + paired significance across all results.

Reads the per-seed *_raw.csv files and emits:
  results/stats_summary.csv      — per (table, model): mean, 95% bootstrap CI on F1/AUC
  results/significance_all.csv   — paired tests (q_plasmanet vs each comparator)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import add_common_args, load_config  # noqa: E402
from src.utils.stats import bootstrap_ci, paired_test  # noqa: E402


def _ci_rows(df, group_cols, metrics=("f1", "auc", "acc"), table=""):
    rows = []
    for keys, g in df.groupby(group_cols):
        rec = {"table": table}
        rec.update(dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))))
        rec["n_seeds"] = len(g)
        for m in metrics:
            if m in g.columns:
                mean, lo, hi = bootstrap_ci(g[m].dropna().to_numpy())
                rec[f"{m}_mean"] = round(mean, 4)
                rec[f"{m}_ci_lo"] = round(lo, 4); rec[f"{m}_ci_hi"] = round(hi, 4)
        rows.append(rec)
    return rows


def run_stats(cfg) -> pd.DataFrame:
    repo = Path(cfg._repo_root)
    R = repo / cfg.get_path("paths.results")

    def rd(name):
        p = R / name
        return pd.read_csv(p) if p.is_file() else None

    ci_rows = []
    bl = rd("baselines_raw.csv")
    if bl is not None:
        ci_rows += _ci_rows(bl, ["model"], table="baselines")
    qm = rd("quantum_main_raw.csv")
    if qm is not None:
        ci_rows += _ci_rows(qm, ["model"], table="quantum_main")
    ab = rd("quantum_ablation_raw.csv")
    if ab is not None:
        ci_rows += _ci_rows(ab, ["qubits", "depth", "entanglement"], table="ablation")
    ex = rd("external_validation.csv")
    if ex is not None and "seed" in ex.columns:
        ci_rows += _ci_rows(ex, ["dataset", "head"], table="external")

    stats = pd.DataFrame(ci_rows)
    stats.to_csv(R / "stats_summary.csv", index=False)

    # Paired tests: q_plasmanet vs cnn (baselines) and vs mlp (quantum_main).
    sig = []
    if qm is not None:
        q = qm[qm.model == "q_plasmanet"].sort_values("seed")["f1"].to_numpy()
        if bl is not None and "cnn" in set(bl.model):
            cnn = bl[bl.model == "cnn"].sort_values("seed")["f1"].to_numpy()
            if len(q) == len(cnn):
                sig.append({"comparison": "q_vs_cnn", **paired_test(q, cnn)})
        if "mlp_param_matched" in set(qm.model):
            mlp = qm[qm.model == "mlp_param_matched"].sort_values("seed")["f1"].to_numpy()
            sig.append({"comparison": "q_vs_mlp", **paired_test(q, mlp)})
    sdf = pd.DataFrame(sig)
    sdf.to_csv(R / "significance_all.csv", index=False)

    print(f"[stats] wrote {R/'stats_summary.csv'} ({len(stats)} rows) and significance_all.csv")
    if not sdf.empty:
        print(sdf.round(4).to_string(index=False))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Consolidate CIs + significance (Q1).")
    add_common_args(p)
    args = p.parse_args()
    run_stats(load_config(args.config or ["configs/base.yaml"]))


if __name__ == "__main__":
    main()
