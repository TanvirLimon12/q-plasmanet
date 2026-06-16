"""Q1 analysis — normalized effective dimension (Abbas et al., Nat. Comput. Sci.
1:403, 2021, "The power of quantum neural networks").

d_n = 2 * ln( (1/m) Σ_i sqrt(det(I_d + γ_n F̂(θ_i))) ) / ln(γ_n),
  γ_n = n / (2π ln n),   F̂ = d · F / mean_θ tr(F),   F = empirical Fisher.

For a softmax classifier p(y|x;θ)=softmax(f(x;θ)), the per-input Fisher is
F_x = J_xᵀ (diag(p) − p pᵀ) J_x with J_x = ∂f/∂θ (K×d); F(θ)=E_x[F_x].

Computed for the re-uploading VQC head vs a parameter-matched classical MLP on the
same features. A higher normalized effective dimension = higher model capacity;
comparing the two characterizes WHY accuracy is at parity on near-separable data.

Output: results/effective_dim.csv  (model, n_params, n_samples, eff_dim, eff_dim_norm)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.precompute import load_features  # noqa: E402
from src.models.classical_heads import mlp_for_param_budget  # noqa: E402
from src.models.quantum_head import build_reupload_head, count_reupload_params  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402


def _flat_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _reinit(model, rng):
    with torch.no_grad():
        for p in _flat_params(model):
            p.copy_(torch.tensor(rng.uniform(-np.pi, np.pi, tuple(p.shape)), dtype=p.dtype))


def _jacobian(model, x):
    """(K, d) Jacobian of logits wrt all params for a single input x."""
    params = _flat_params(model)
    logits = model(x.unsqueeze(0))[0]                 # (K,)
    rows = []
    for k in range(logits.shape[0]):
        grads = torch.autograd.grad(logits[k], params, retain_graph=(k < logits.shape[0] - 1),
                                    create_graph=False, allow_unused=True)
        rows.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                               for g, p in zip(grads, params)]))
    return torch.stack(rows)                           # (K, d)


def _fisher(model, X):
    d = sum(p.numel() for p in _flat_params(model))
    F = np.zeros((d, d))
    for x in X:
        J = _jacobian(model, x)                        # (K,d)
        with torch.no_grad():
            p = torch.softmax(model(x.unsqueeze(0))[0], dim=0).numpy()
        A = np.diag(p) - np.outer(p, p)                # (K,K)
        Jn = J.detach().numpy()
        F += Jn.T @ A @ Jn
    return F / len(X)


def effective_dimension(build_model, X, n_values, m_theta=30, seed=0):
    rng = np.random.default_rng(seed)
    fishers, traces = [], []
    for _ in range(m_theta):
        model = build_model()
        _reinit(model, rng)
        F = _fisher(model, X)
        fishers.append(F); traces.append(np.trace(F))
    d = fishers[0].shape[0]
    norm = d / (np.mean(traces) + 1e-12)
    Fhat = [norm * F for F in fishers]

    out = {}
    for n in n_values:
        gamma = n / (2 * np.pi * np.log(n))
        terms = []
        for F in Fhat:
            sign, logdet = np.linalg.slogdet(np.eye(d) + gamma * F)
            terms.append(0.5 * logdet)                 # ln sqrt(det) = 0.5 logdet
        m = np.max(terms)
        log_mean = m + np.log(np.mean(np.exp(np.array(terms) - m)))   # logsumexp-style
        out[n] = float(2 * log_mean / np.log(gamma))
    return out, d


def run_effective_dim(cfg, seed=0, n_data=64, m_theta=30,
                      n_values=(1000, 10000, 100000, 1000000)) -> pd.DataFrame:
    repo = Path(cfg._repo_root)
    results = repo / cfg.get_path("paths.results"); results.mkdir(parents=True, exist_ok=True)
    backbone = cfg.get_path("feature_backbone") or "cnn"
    X, _ = load_features(cfg, backbone, 24, seed, "train")
    if X is None:
        raise FileNotFoundError("need d24 features; run precompute --from-classifier cnn.")
    X = torch.tensor(X[:n_data], dtype=torch.float32)

    target = count_reupload_params(8, 3)["total"]
    builders = {
        "q_plasmanet": lambda: build_reupload_head(8, 3, "circular", "lightning.qubit",
                                                   "adjoint", verify=False)[0],
        "mlp_param_matched": lambda: mlp_for_param_budget(24, target, 2),
    }
    rows = []
    for name, build in builders.items():
        ed, d = effective_dimension(build, X, n_values, m_theta=m_theta, seed=seed)
        for n, val in ed.items():
            rows.append({"model": name, "n_params": d, "n_data": int(n),
                         "eff_dim": val, "eff_dim_norm": val / d})
        print(f"[eff_dim] {name}: d={d} eff_dim(n=1e6)={ed[max(n_values)]:.2f} "
              f"norm={ed[max(n_values)]/d:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(results / "effective_dim.csv", index=False)
    print(f"[eff_dim] wrote {results/'effective_dim.csv'}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Normalized effective dimension (Abbas 2021).")
    add_common_args(p)
    p.add_argument("--n-data", type=int, default=64)
    p.add_argument("--m-theta", type=int, default=30)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml", "configs/quantum.yaml"])
    cfg["feature_backbone"] = "cnn"
    if args.smoke:
        run_effective_dim(cfg, n_data=16, m_theta=5, n_values=(1000, 100000))
    else:
        run_effective_dim(cfg, n_data=args.n_data, m_theta=args.m_theta)


if __name__ == "__main__":
    main()
