"""Phase 6 — NISQ feasibility: re-evaluate the trained quantum head under finite
shots and hardware noise. Inference only (no retraining).

Two studies, deliberately separated so each effect is isolated (reviewer-clean):
  - Shot noise: analytic-trained weights re-evaluated with a sampling device at
    {128, 512, 1024, 4096} shots (noiseless channels).
  - Hardware noise: analytic (shots=None) density-matrix evaluation with one
    channel active at a time — depolarizing, amplitude damping, phase damping,
    readout error.

A third "calibrated" study maps published device calibration (T1/T2, gate times,
gate-error rates, readout error) to per-gate ``default.mixed`` channels — i.e.
calibration-informed noise without Qiskit or live hardware (cite the source).

Implementation is single-framework PennyLane (``default.mixed`` density matrix +
sampling); no Qiskit dependency, fully reproducible.

Compute warning: noisy/shot simulation is 10-100x slower than statevector. Run
the full grid on ONE representative config; checkpoint partial results to CSV so a
crash does not lose the run.

Deliverables:
  results/nisq.csv   (Table 6 + noise-sensitivity plot data)
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
from src.eval.metrics import compute_metrics  # noqa: E402
from src.features.precompute import load_features  # noqa: E402
from src.models.quantum_head import entangle_pairs, reupload_input_dim  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402

_note = (
    "Single framework: PennyLane default.mixed. The 'calibrated' study maps device "
    "calibration (T1/T2/gate-error/readout) to per-gate channels — calibration-informed "
    "noise without Qiskit or live hardware; cite the calibration source in the manuscript."
)


def calibrated_rates(cal: dict) -> dict:
    """Map device calibration to per-gate PennyLane channel rates.

    amplitude damping γ = 1−e^(−t/T1); phase damping γ = 1−e^(−t/T2φ) with
    1/T2φ = 1/T2 − 1/(2 T1); depolarizing = gate error; readout = measurement error.
    """
    t1, t2 = float(cal["t1_us"]), float(cal["t2_us"])
    out = {}
    for tag, tg_ns in (("1q", cal["gate_time_1q_ns"]), ("2q", cal["gate_time_2q_ns"])):
        t_us = float(tg_ns) / 1000.0
        amp = 1.0 - np.exp(-t_us / t1)
        inv_t2phi = max(1e-9, 1.0 / t2 - 1.0 / (2.0 * t1))
        phase = 1.0 - np.exp(-t_us * inv_t2phi)
        out[tag] = {"amp": float(amp), "phase": float(phase),
                    "depol": float(cal[f"gate_error_{tag}"])}
    out["readout"] = float(cal["readout_error"])
    return out


def load_trained_vqc(cfg, seed: int, backbone: str):
    """Return (state_dict_np, hparams) from the Phase-4 main checkpoint.

    state_dict_np holds numpy arrays: 'weights', 'fc_w', 'fc_b', and (reupload only)
    'scaling'.
    """
    repo_root = Path(cfg._repo_root)
    ckpt = repo_root / cfg.get_path("paths.checkpoints") / f"quantum_main_{backbone}_seed{seed}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing VQC checkpoint {ckpt}; run Phase 4 main first.")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    out = {
        "weights": sd["qlayer.weights"].cpu().numpy(),
        "fc_w": sd["fc.weight"].cpu().numpy(),
        "fc_b": sd["fc.bias"].cpu().numpy(),
    }
    if "qlayer.scaling" in sd:                      # reupload head
        out["scaling"] = sd["qlayer.scaling"].cpu().numpy()
    return out, ck["hparams"]


def make_noisy_qnode(n, depth, entanglement, encoding, shots, noise_type, noise_p):
    """Density-matrix qnode replicating the trained ansatz, with one channel active."""
    dev = qml.device("default.mixed", wires=n, shots=shots)
    pairs = entangle_pairs(n, entanglement)

    def channel(wires):
        if noise_type == "depolarizing":
            for w in wires:
                qml.DepolarizingChannel(noise_p, wires=w)
        elif noise_type == "amplitude_damping":
            for w in wires:
                qml.AmplitudeDamping(noise_p, wires=w)
        elif noise_type == "phase_damping":
            for w in wires:
                qml.PhaseDamping(noise_p, wires=w)
        # 'readout_error' and 'none' insert no gate-level channel.

    @qml.qnode(dev)
    def circuit(inputs, weights):
        if encoding == "angle":
            qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
        else:
            qml.AmplitudeEmbedding(inputs, wires=range(n), normalize=True, pad_with=0.0)
        for layer in range(depth):
            for w in range(n):
                qml.RX(weights[layer, w, 0], wires=w)
                qml.RY(weights[layer, w, 1], wires=w)
                qml.RZ(weights[layer, w, 2], wires=w)
            channel(range(n))
            for a, b in pairs:
                qml.CNOT(wires=[a, b])
            channel(range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circuit


def make_noisy_reupload_qnode(n, layers, entanglement, shots, noise_type, noise_p):
    """Noisy density-matrix qnode for the data re-uploading head (Rot encoding +
    trainable scaling, <Z>+<ZZ> readout), one channel active."""
    dev = qml.device("default.mixed", wires=n, shots=shots)
    ent = entangle_pairs(n, entanglement)
    zz = entangle_pairs(n, "circular")

    def channel(wires):
        if noise_type == "depolarizing":
            for w in wires:
                qml.DepolarizingChannel(noise_p, wires=w)
        elif noise_type == "amplitude_damping":
            for w in wires:
                qml.AmplitudeDamping(noise_p, wires=w)
        elif noise_type == "phase_damping":
            for w in wires:
                qml.PhaseDamping(noise_p, wires=w)

    @qml.qnode(dev)
    def circuit(inputs, scaling, weights):
        for l in range(layers):
            f = scaling[l] * inputs
            for q in range(n):
                qml.Rot(f[3 * q], f[3 * q + 1], f[3 * q + 2], wires=q)
            for q in range(n):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
            channel(range(n))
            for a, b in ent:
                qml.CNOT(wires=[a, b])
            channel(range(n))
        obs = [qml.expval(qml.PauliZ(i)) for i in range(n)]
        obs += [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in zz]
        return obs

    return circuit


def _ch1(rates, w):
    qml.AmplitudeDamping(rates["1q"]["amp"], wires=w)
    qml.PhaseDamping(rates["1q"]["phase"], wires=w)
    qml.DepolarizingChannel(rates["1q"]["depol"], wires=w)


def _ch2(rates, a, b):
    for w in (a, b):
        qml.AmplitudeDamping(rates["2q"]["amp"], wires=w)
        qml.PhaseDamping(rates["2q"]["phase"], wires=w)
        qml.DepolarizingChannel(rates["2q"]["depol"], wires=w)


def make_calibrated_qnode(n, depth, entanglement, encoding, shots, rates):
    """Calibration-informed (T1/T2/gate-error) noise on the angle ansatz."""
    dev = qml.device("default.mixed", wires=n, shots=shots)
    pairs = entangle_pairs(n, entanglement)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        if encoding == "angle":
            qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
        else:
            qml.AmplitudeEmbedding(inputs, wires=range(n), normalize=True, pad_with=0.0)
        for w in range(n):
            _ch1(rates, w)
        for layer in range(depth):
            for w in range(n):
                qml.RX(weights[layer, w, 0], wires=w)
                qml.RY(weights[layer, w, 1], wires=w)
                qml.RZ(weights[layer, w, 2], wires=w)
                _ch1(rates, w)
            for a, b in pairs:
                qml.CNOT(wires=[a, b]); _ch2(rates, a, b)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circuit


def make_calibrated_reupload_qnode(n, layers, entanglement, shots, rates):
    """Calibration-informed noise on the data re-uploading head."""
    dev = qml.device("default.mixed", wires=n, shots=shots)
    ent = entangle_pairs(n, entanglement)
    zz = entangle_pairs(n, "circular")

    @qml.qnode(dev)
    def circuit(inputs, scaling, weights):
        for l in range(layers):
            f = scaling[l] * inputs
            for q in range(n):
                qml.Rot(f[3 * q], f[3 * q + 1], f[3 * q + 2], wires=q); _ch1(rates, q)
            for q in range(n):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q); _ch1(rates, q)
            for a, b in ent:
                qml.CNOT(wires=[a, b]); _ch2(rates, a, b)
        obs = [qml.expval(qml.PauliZ(i)) for i in range(n)]
        obs += [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in zz]
        return obs

    return circuit


def calibrated_predict(X, sd, hp, shots, rates) -> np.ndarray:
    n = hp["qubits"]
    if hp["encoding"] == "reupload":
        circuit = make_calibrated_reupload_qnode(n, hp["depth"], hp["entanglement"], shots, rates)
        evalf = lambda x: np.asarray(circuit(x, sd["scaling"], sd["weights"]), dtype=float)
    else:
        circuit = make_calibrated_qnode(n, hp["depth"], hp["entanglement"], hp["encoding"], shots, rates)
        evalf = lambda x: np.asarray(circuit(x, sd["weights"]), dtype=float)
    ro = 1.0 - 2.0 * rates["readout"]
    fc_w, fc_b = sd["fc_w"], sd["fc_b"]
    probs = []
    for x in X:
        z = evalf(x) * ro
        logits = fc_w @ z + fc_b
        logits = logits - logits.max()
        e = np.exp(logits)
        probs.append(float(e[1] / e.sum()))
    return np.asarray(probs)


def noisy_predict(X, sd, hp, shots, noise_type, noise_p) -> np.ndarray:
    """Per-sample noisy evaluation -> probability of class 1. Dispatches by encoding."""
    n = hp["qubits"]
    ro_scale = (1.0 - 2.0 * noise_p) if noise_type == "readout_error" else 1.0
    if hp["encoding"] == "reupload":
        circuit = make_noisy_reupload_qnode(n, hp["depth"], hp["entanglement"],
                                            shots, noise_type, noise_p)
        evalf = lambda x: np.asarray(circuit(x, sd["scaling"], sd["weights"]), dtype=float)
    else:
        circuit = make_noisy_qnode(n, hp["depth"], hp["entanglement"], hp["encoding"],
                                   shots, noise_type, noise_p)
        evalf = lambda x: np.asarray(circuit(x, sd["weights"]), dtype=float)

    fc_w, fc_b = sd["fc_w"], sd["fc_b"]
    probs = []
    for x in X:
        z = evalf(x) * ro_scale
        logits = fc_w @ z + fc_b
        logits = logits - logits.max()
        e = np.exp(logits)
        probs.append(float(e[1] / e.sum()))
    return np.asarray(probs)


def run_nisq(cfg, seeds=None, smoke: bool = False, limit: int | None = None,
             out_name: str = "nisq.csv") -> pd.DataFrame:
    repo_root = Path(cfg._repo_root)
    results_dir = repo_root / cfg.get_path("paths.results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / out_name
    backbone = cfg.get_path("feature_backbone") or "resnet18"
    checkpoint_partial = bool(cfg.get_path("checkpoint_partial", True))

    if smoke:
        seeds = seeds or cfg.get_path("smoke.seeds")
        shots_list = [128]
        noise_grid = [("none", 0.0), ("depolarizing", 0.05)]
        limit = limit if limit is not None else 24
    else:
        seeds = seeds or cfg.get_path("seeds")
        shots_list = cfg.get_path("shots") or [128, 512, 1024, 4096]
        nm = cfg.get_path("noise_models") or {}
        noise_grid = []
        for ntype, spec in nm.items():
            if not spec.get("enabled", False):
                continue
            if ntype == "readout_error":
                p = spec.get("p01", 0.02)
                noise_grid.append((ntype, p))
            else:
                key = "p" if ntype == "depolarizing" else "gamma"
                for val in spec.get(key, []):
                    noise_grid.append((ntype, val))

    rows = []
    for seed in seeds:
        sd, hp = load_trained_vqc(cfg, seed, backbone)
        feat_dim = reupload_input_dim(hp["qubits"]) if hp["encoding"] == "reupload" else hp["qubits"]
        X, y = load_features(cfg, backbone, feat_dim, seed, "test")
        if X is None:
            raise FileNotFoundError(f"no d{feat_dim} test features for seed {seed}.")
        if limit is not None:
            X, y = X[:limit], y[:limit]

        # --- Shot-noise study: noiseless channels, sweep shots ---
        for shots in shots_list:
            p = noisy_predict(X, sd, hp, shots, "none", 0.0)
            m = compute_metrics(y, p)
            rows.append({"seed": seed, "study": "shots", "shots": shots,
                         "noise_type": "none", "noise_p": 0.0, **m,
                         **{f"cfg_{k}": hp[k] for k in ("qubits", "depth", "entanglement", "encoding")}})
            if checkpoint_partial:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[nisq] seed{seed} shots={shots} noise=none -> acc={m['acc']:.3f} auc={m['auc']:.3f}")

        # --- Hardware-noise study: analytic (shots=None), one channel at a time ---
        for ntype, p_val in noise_grid:
            if ntype == "none":
                continue
            p = noisy_predict(X, sd, hp, None, ntype, p_val)
            m = compute_metrics(y, p)
            rows.append({"seed": seed, "study": "noise", "shots": None,
                         "noise_type": ntype, "noise_p": p_val, **m,
                         **{f"cfg_{k}": hp[k] for k in ("qubits", "depth", "entanglement", "encoding")}})
            if checkpoint_partial:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[nisq] seed{seed} {ntype}={p_val} (analytic) -> acc={m['acc']:.3f} auc={m['auc']:.3f}")

        # --- Calibration-informed study: all channels per gate, device-derived rates ---
        cal = cfg.get_path("calibration")
        if cal and cal.get("enabled", False):
            rates = calibrated_rates(cal)
            p = calibrated_predict(X, sd, hp, None, rates)
            m = compute_metrics(y, p)
            rows.append({"seed": seed, "study": "calibrated", "shots": None,
                         "noise_type": f"calibrated:{cal.get('device','device')}", "noise_p": np.nan, **m,
                         **{f"cfg_{k}": hp[k] for k in ("qubits", "depth", "entanglement", "encoding")}})
            if checkpoint_partial:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[nisq] seed{seed} calibrated({cal.get('device')}) "
                  f"amp1q={rates['1q']['amp']:.2e} phase1q={rates['1q']['phase']:.2e} "
                  f"depol2q={rates['2q']['depol']:.2e} -> acc={m['acc']:.3f} auc={m['auc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n[nisq] wrote {out_csv} ({len(df)} rows)")
    print(f"[nisq] note: {_note}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="NISQ shot/noise evaluation (Phase 6).")
    add_common_args(p)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap test samples (8-qubit density-matrix sim is heavy).")
    args = p.parse_args()
    configs = args.config or ["configs/backbones.yaml", "configs/quantum.yaml", "configs/noise.yaml"]
    cfg = load_config(configs)
    run_nisq(cfg, seeds=args.seeds, smoke=args.smoke, limit=args.limit)


if __name__ == "__main__":
    main()
