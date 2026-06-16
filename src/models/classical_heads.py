"""Phase 3 — classification heads + backbone/head assembly.

The parameter-matched MLP head is the key fair comparator against the quantum
head: ``mlp_for_param_budget`` sizes a single-hidden-layer MLP so its trainable
parameter count is close to a target (the quantum head's count, finalized in
Phase 4).
"""
from __future__ import annotations

import torch.nn as nn

from src.models.backbones import build_backbone, freeze


def count_params(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def build_head(kind: str, in_dim: int, n_classes: int = 2, cfg=None) -> nn.Module:
    kind = kind.lower()
    if kind == "linear":
        return nn.Linear(in_dim, n_classes)
    if kind == "mlp":
        hidden = (cfg.get_path("backbones.heads.mlp.hidden") if cfg else None) or [32, 16]
        dropout = (cfg.get_path("backbones.heads.mlp.dropout") if cfg else None) or 0.1
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        return nn.Sequential(*layers)
    raise ValueError(f"unknown head {kind!r}")


def mlp_for_param_budget(in_dim: int, target_params: int, n_classes: int = 2) -> nn.Module:
    """Single-hidden-layer MLP whose param count ≈ target_params.

    params ≈ in_dim*h + h + h*n_classes + n_classes  =>  h ≈ (target - n_classes)/(in_dim + 1 + n_classes)
    """
    denom = in_dim + 1 + n_classes
    h = max(1, round((target_params - n_classes) / denom))
    return nn.Sequential(nn.Linear(in_dim, h), nn.ReLU(), nn.Linear(h, n_classes))


class ClassicalClassifier(nn.Module):
    """Backbone + head. Frozen backbone trains the head only (default for pretrained)."""

    def __init__(self, backbone: nn.Module, head: nn.Module, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            freeze(self.backbone)

    def forward(self, x):
        if self.freeze_backbone:
            self.backbone.eval()
        feats = self.backbone(x)
        return self.head(feats)


def build_classifier(model_name: str, cfg, n_classes: int = 2, pretrained: bool | None = None) -> ClassicalClassifier:
    """Compose a named baseline. 'cnn_mlp' = custom CNN + parameter-matched MLP head."""
    name = model_name.lower()
    freeze_cfg = bool(cfg.get_path("backbones.train.freeze_backbone")) if cfg else False

    if name == "cnn_mlp":
        backbone, feat = build_backbone("cnn", cfg, pretrained=False)
        target = int((cfg.get_path("quantum_head_param_target") if cfg else None) or 48)
        head = mlp_for_param_budget(feat, target_params=target, n_classes=n_classes)
        return ClassicalClassifier(backbone, head, freeze_backbone=False)

    backbone, feat = build_backbone(name, cfg, pretrained=pretrained)
    # Custom CNN is always trainable; pretrained backbones honor the freeze flag.
    do_freeze = freeze_cfg and name not in ("cnn", "smallcnn")
    head = build_head("linear", feat, n_classes=n_classes, cfg=cfg)
    return ClassicalClassifier(backbone, head, freeze_backbone=do_freeze)
