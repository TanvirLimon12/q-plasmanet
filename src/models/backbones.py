"""Phase 3 — feature-extractor backbones.

Each backbone maps an image batch (B,3,H,W) to a feature vector (B, feat_dim).
Pretrained torchvision weights are used when requested and available; if the
weights cannot be fetched (e.g. offline), it falls back to random init with a
warning rather than crashing.

Returns (module, feat_dim). The classifier/head is attached separately.
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """Custom compact CNN. Always trainable (it learns the features)."""

    def __init__(self, in_ch: int = 3, feat_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)


def _tv_weights(weights_enum, pretrained: bool):
    if not pretrained:
        return None
    try:
        return weights_enum.DEFAULT
    except Exception as e:  # pragma: no cover
        warnings.warn(f"pretrained weights unavailable ({e}); using random init")
        return None


class _TVBackbone(nn.Module):
    """Wrap a torchvision model as a feature extractor (classifier stripped)."""

    def __init__(self, body: nn.Module, feat_dim: int):
        super().__init__()
        self.body = body
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.body(x).flatten(1)


def build_backbone(name: str, cfg=None, pretrained: bool | None = None) -> tuple[nn.Module, int]:
    name = name.lower()

    if name in ("cnn", "smallcnn"):
        feat = int((cfg.get_path("backbones.backbones.cnn.feature_dim") if cfg else None) or 256)
        m = SmallCNN(feat_dim=feat)
        return m, m.feat_dim

    want_pre = pretrained if pretrained is not None else True

    if name in ("resnet18", "resnet"):
        w = _tv_weights(models.ResNet18_Weights, want_pre)
        net = models.resnet18(weights=w)
        feat = net.fc.in_features            # 512
        net.fc = nn.Identity()
        return _TVBackbone(net, feat), feat

    if name in ("mobilenet_v2", "mobilenetv2"):
        w = _tv_weights(models.MobileNet_V2_Weights, want_pre)
        net = models.mobilenet_v2(weights=w)
        feat = net.classifier[-1].in_features  # 1280
        net.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1) if False else nn.Identity())
        # mobilenet_v2.forward already global-pools features; strip classifier only
        net.classifier = nn.Identity()
        return _TVBackbone(net, feat), feat

    if name in ("efficientnet_b0", "efficientnetb0", "effnet_b0"):
        w = _tv_weights(models.EfficientNet_B0_Weights, want_pre)
        net = models.efficientnet_b0(weights=w)
        feat = net.classifier[-1].in_features  # 1280
        net.classifier = nn.Identity()
        return _TVBackbone(net, feat), feat

    raise ValueError(f"unknown backbone {name!r}")


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
