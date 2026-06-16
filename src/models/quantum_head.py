"""Phase 4 — PennyLane variational quantum head.

Angle encoding (main; amplitude optional), an RX/RY/RZ + CNOT ansatz with
configurable entanglement, Pauli-Z readout, wrapped as a ``qml.qnn.TorchLayer``
so it differentiates inside an ``nn.Module``.

Gradients: analytic. Default ``diff_method`` is the device's exact adjoint on
``lightning.qubit``; ``parameter-shift`` is available for hardware-faithful runs
(set ``diff_method='parameter-shift'``). Training uses statevector simulation —
the shot/noise study lives in Phase 6 (PennyLane default.mixed).
"""
from __future__ import annotations

import warnings

import pennylane as qml
import torch
import torch.nn as nn


def entangle_pairs(n: int, kind: str) -> list[tuple[int, int]]:
    kind = kind.lower()
    if kind == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "circular":
        pairs = [(i, i + 1) for i in range(n - 1)]
        if n > 2:
            pairs.append((n - 1, 0))
        return pairs
    if kind == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"unknown entanglement {kind!r}")


def input_dim(n_qubits: int, encoding: str) -> int:
    return n_qubits if encoding.lower() == "angle" else 2 ** n_qubits


def _make_qnode(n, depth, entanglement, encoding, device_name, diff_method):
    dev = qml.device(device_name, wires=n)
    pairs = entangle_pairs(n, entanglement)
    enc = encoding.lower()

    def circuit(inputs, weights):
        if enc == "angle":
            qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
        else:
            qml.AmplitudeEmbedding(inputs, wires=range(n), normalize=True, pad_with=0.0)
        for layer in range(depth):
            for w in range(n):
                qml.RX(weights[layer, w, 0], wires=w)
                qml.RY(weights[layer, w, 1], wires=w)
                qml.RZ(weights[layer, w, 2], wires=w)
            for a, b in pairs:
                qml.CNOT(wires=[a, b])
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)


class QuantumHead(nn.Module):
    """VQC (TorchLayer) + classical linear readout -> class logits."""

    def __init__(self, n_qubits, depth, entanglement, encoding, qnode, n_classes=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.encoding = encoding
        self.in_dim = input_dim(n_qubits, encoding)
        weight_shapes = {"weights": (depth, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        z = self.qlayer(x)                  # (B, n_qubits) expectation values in [-1,1]
        return self.fc(z)


def count_quantum_params(n_qubits: int, depth: int, n_classes: int = 2) -> dict:
    ansatz = depth * n_qubits * 3
    readout = n_qubits * n_classes + n_classes
    return {"ansatz": ansatz, "readout": readout, "total": ansatz + readout}


def build_quantum_head(
    n_qubits: int = 4,
    depth: int = 1,
    entanglement: str = "linear",
    encoding: str = "angle",
    device_name: str = "lightning.qubit",
    diff_method: str = "adjoint",
    n_classes: int = 2,
    verify: bool = True,
) -> tuple[QuantumHead, dict]:
    """Build a quantum head. If the requested device/diff_method fails a dummy
    forward (e.g. adjoint+batching unsupported), fall back to
    default.qubit / backprop with a warning."""
    def _try(devn, diff):
        qn = _make_qnode(n_qubits, depth, entanglement, encoding, devn, diff)
        head = QuantumHead(n_qubits, depth, entanglement, encoding, qn, n_classes)
        if verify:
            x = torch.zeros(2, input_dim(n_qubits, encoding))
            out = head(x)                   # smoke forward + backward (grad path)
            out.sum().backward()
            head.zero_grad()
        return head

    used_device, used_diff = device_name, diff_method
    try:
        head = _try(device_name, diff_method)
    except Exception as e:
        warnings.warn(
            f"quantum head ({device_name}/{diff_method}) failed verify ({e}); "
            f"falling back to default.qubit/backprop"
        )
        used_device, used_diff = "default.qubit", "backprop"
        head = _try(used_device, used_diff)

    info = {
        "n_qubits": n_qubits, "depth": depth, "entanglement": entanglement,
        "encoding": encoding, "device": used_device, "diff_method": used_diff,
        "in_dim": input_dim(n_qubits, encoding),
        **{f"params_{k}": v for k, v in count_quantum_params(n_qubits, depth, n_classes).items()},
    }
    return head, info


# ---------------------------------------------------------------------------
# Data re-uploading head (research upgrade): interleaved encode/trainable layers
# with trainable input scaling (Schuld 2021, PRA 103:032430) and per-qubit Rot
# encoding so each upload ingests 3*n_qubits features (more information than plain
# angle encoding). Richer readout: <Z_i> + <Z_i Z_j> correlators. References:
# Perez-Salinas et al., Quantum 4:226 (2020); Schuld et al., PRA 103:032430 (2021).
# ---------------------------------------------------------------------------

def reupload_input_dim(n_qubits: int) -> int:
    return 3 * n_qubits                     # Rot = 3 angles per qubit per upload


def reupload_readout_dim(n_qubits: int) -> int:
    pairs = entangle_pairs(n_qubits, "circular")
    return n_qubits + len(pairs)            # <Z_i> + <Z_i Z_j>


def count_reupload_params(n_qubits: int, layers: int, n_classes: int = 2) -> dict:
    scaling = layers * 3 * n_qubits
    ansatz = layers * n_qubits * 3
    readout = reupload_readout_dim(n_qubits) * n_classes + n_classes
    return {"scaling": scaling, "ansatz": ansatz, "readout": readout,
            "total": scaling + ansatz + readout}


def _make_reupload_qnode(n, layers, entanglement, device_name, diff_method):
    dev = qml.device(device_name, wires=n)
    ent = entangle_pairs(n, entanglement)
    zz = entangle_pairs(n, "circular")

    def circuit(inputs, scaling, weights):
        # inputs:(3n,)  scaling:(layers,3n)  weights:(layers,n,3)
        for l in range(layers):
            f = scaling[l] * inputs                       # trainable frequency scaling
            for q in range(n):                            # index last axis (batch-safe)
                qml.Rot(f[..., 3 * q], f[..., 3 * q + 1], f[..., 3 * q + 2], wires=q)
            for q in range(n):
                qml.Rot(weights[l, q, 0], weights[l, q, 1], weights[l, q, 2], wires=q)
            for a, b in ent:
                qml.CNOT(wires=[a, b])
        obs = [qml.expval(qml.PauliZ(i)) for i in range(n)]
        obs += [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in zz]
        return obs

    return qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)


class ReuploadQuantumHead(nn.Module):
    """Data re-uploading VQC + correlator readout -> class logits."""

    def __init__(self, n_qubits, layers, entanglement, qnode, n_classes=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.encoding = "reupload"
        self.in_dim = reupload_input_dim(n_qubits)
        weight_shapes = {"scaling": (layers, 3 * n_qubits), "weights": (layers, n_qubits, 3)}
        init = {
            "scaling": lambda t: nn.init.normal_(t, mean=1.0, std=0.1),   # ~pass-through
            "weights": lambda t: nn.init.normal_(t, mean=0.0, std=0.1),
        }
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes, init_method=init)
        self.fc = nn.Linear(reupload_readout_dim(n_qubits), n_classes)

    def forward(self, x):
        return self.fc(self.qlayer(x))


def build_reupload_head(
    n_qubits: int = 8,
    layers: int = 3,
    entanglement: str = "circular",
    device_name: str = "lightning.qubit",
    diff_method: str = "adjoint",
    n_classes: int = 2,
    verify: bool = True,
) -> tuple[ReuploadQuantumHead, dict]:
    def _try(devn, diff):
        qn = _make_reupload_qnode(n_qubits, layers, entanglement, devn, diff)
        head = ReuploadQuantumHead(n_qubits, layers, entanglement, qn, n_classes)
        if verify:
            out = head(torch.zeros(2, reupload_input_dim(n_qubits)))
            out.sum().backward()
            head.zero_grad()
        return head

    used_device, used_diff = device_name, diff_method
    try:
        head = _try(device_name, diff_method)
    except Exception as e:
        warnings.warn(f"reupload head ({device_name}/{diff_method}) failed verify ({e}); "
                      f"falling back to default.qubit/backprop")
        used_device, used_diff = "default.qubit", "backprop"
        head = _try(used_device, used_diff)

    info = {
        "n_qubits": n_qubits, "depth": layers, "entanglement": entanglement,
        "encoding": "reupload", "device": used_device, "diff_method": used_diff,
        "in_dim": reupload_input_dim(n_qubits),
        **{f"params_{k}": v for k, v in count_reupload_params(n_qubits, layers, n_classes).items()},
    }
    return head, info
