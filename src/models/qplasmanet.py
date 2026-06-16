"""Backbone + head assembly and inference loaders.

`QPlasmaNet` composes the frozen backbone, the fitted PCA+scaler transform, and
the trained VQC head into a single image -> logits module. Used by Phases 5-6 to
evaluate the full pipeline on new/corrupted/noisy inputs (inference only — the
sklearn transform breaks autograd, which is fine here).

Loaders rebuild the exact artifacts written by Phase 4:
  build_qplasmanet_infer  — backbone(.pt) + transform(.joblib) + VQC(.pt)
  build_classical_infer   — self-contained classical checkpoint
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.precompute import load_transform  # noqa: E402
from src.models.backbones import build_backbone, freeze  # noqa: E402
from src.models.classical_heads import build_classifier  # noqa: E402
from src.models.quantum_head import build_quantum_head, build_reupload_head, reupload_input_dim  # noqa: E402


class QPlasmaNet(nn.Module):
    """Frozen backbone -> fitted PCA/scaler -> VQC head. Image batch -> class logits."""

    def __init__(self, backbone: nn.Module, transform, quantum_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.transform = transform           # fitted sklearn Pipeline (numpy)
        self.quantum_head = quantum_head

    @torch.no_grad()
    def forward(self, x):
        self.backbone.eval()
        feats = self.backbone(x).cpu().numpy()
        z = self.transform.transform(feats).astype(np.float32)
        zt = torch.as_tensor(z, device=next(self.quantum_head.parameters()).device)
        return self.quantum_head(zt)


def build_qplasmanet_infer(cfg, seed: int, backbone_name: str | None = None, device=None) -> QPlasmaNet:
    repo_root = Path(cfg._repo_root)
    proc = repo_root / cfg.get_path("data.processed_root")
    ckpt_dir = repo_root / cfg.get_path("paths.checkpoints")
    subset = cfg.get_path("data.primary_subset")
    backbone_name = backbone_name or (cfg.get_path("feature_backbone") or "resnet18")
    # Quantum pipeline runs on CPU (lightning.qubit / TorchLayer don't support MPS);
    # backbone inference on CPU is fine for the modest eval sets.
    device = torch.device("cpu")

    # VQC checkpoint (holds qubits/depth/entanglement/encoding in its hparams).
    ckpt_path = ckpt_dir / f"quantum_main_{backbone_name}_seed{seed}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing VQC checkpoint {ckpt_path}; run Phase 4 main first.")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ck["hparams"]

    # Backbone: rebuild arch, load saved frozen weights. Prefer the per-seed
    # trained backbone (from precompute --from-classifier) when present.
    backbone, _ = build_backbone(backbone_name, cfg, pretrained=False)
    bfile_seed = proc / f"backbone_{subset}_{backbone_name}_seed{seed}.pt"
    bfile = bfile_seed if bfile_seed.is_file() else proc / f"backbone_{subset}_{backbone_name}.pt"
    if bfile.is_file():
        backbone.load_state_dict(torch.load(bfile, map_location=device))
    freeze(backbone)
    backbone.to(device)

    # Feature width depends on encoding: reupload ingests 3*qubits features.
    feat_dim = reupload_input_dim(hp["qubits"]) if hp["encoding"] == "reupload" else hp["qubits"]
    transform = load_transform(cfg, backbone_name, feat_dim, seed)
    if transform is None:
        raise FileNotFoundError(f"missing fitted transform for d{feat_dim} seed {seed}.")

    # VQC head (angle or data re-uploading).
    if hp["encoding"] == "reupload":
        head, _ = build_reupload_head(
            hp["qubits"], hp["depth"], hp["entanglement"],
            device_name=hp.get("device", "lightning.qubit"),
            diff_method=hp.get("diff_method", "adjoint"), verify=False,
        )
    else:
        head, _ = build_quantum_head(
            hp["qubits"], hp["depth"], hp["entanglement"], hp["encoding"],
            device_name=hp.get("device", "lightning.qubit"),
            diff_method=hp.get("diff_method", "adjoint"), verify=False,
        )
    head.load_state_dict(ck["state_dict"])
    head.to(device).eval()
    return QPlasmaNet(backbone, transform, head)


def build_classical_infer(cfg, model_name: str, seed: int, device=None) -> nn.Module:
    repo_root = Path(cfg._repo_root)
    ckpt_dir = repo_root / cfg.get_path("paths.checkpoints")
    subset = cfg.get_path("data.primary_subset")
    device = device or torch.device("cpu")

    ckpt_path = ckpt_dir / f"{subset}_{model_name}_seed{seed}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"missing classical checkpoint {ckpt_path}; run Phase 3 first.")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_classifier(model_name, cfg, n_classes=2, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model
