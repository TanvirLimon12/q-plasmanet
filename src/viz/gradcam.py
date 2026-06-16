"""Phase 9 — Grad-CAM on the cnn backbone (Fig 6).

Highlights which image regions drive the plasma/non-plasma decision. Hooks the
last conv layer of the SmallCNN backbone, back-props the predicted-class score,
and overlays the class-activation map on sample test cells.

  python -m src.viz.gradcam --config configs/backbones.yaml --seed 0 --n 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import build_transforms, get_normalization  # noqa: E402
from src.models.qplasmanet import build_classical_infer  # noqa: E402
from src.train.engine import get_device  # noqa: E402
from src.utils.config import add_common_args, load_config  # noqa: E402


def _last_conv(module: nn.Module) -> nn.Conv2d:
    conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
    if conv is None:
        raise ValueError("no Conv2d found in backbone")
    return conv


def _denorm(x, cfg):
    mean, std = get_normalization(cfg)
    t = x.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def gradcam(cfg, model_name: str = "cnn", seed: int = 0, n: int = 6):
    repo = Path(cfg._repo_root)
    figures = repo / cfg.get_path("paths.figures"); figures.mkdir(parents=True, exist_ok=True)
    splits = repo / cfg.get_path("paths.splits")
    device = get_device()

    model = build_classical_infer(cfg, model_name, seed, device=device)
    target = _last_conv(model.backbone)
    acts, grads = {}, {}
    target.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o.detach()))
    target.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0].detach()))

    df = pd.read_csv(splits / f"seed{seed}_test.csv").sample(min(n, 10**6), random_state=seed)
    tfm = build_transforms(cfg, train=False)
    size = int(cfg.get_path("image.size"))

    disp = 256                                           # high-res display (crisper than 64px input)
    fig, axes = plt.subplots(2, len(df), figsize=(2.6 * len(df), 5.2), squeeze=False)
    for j, (_, row) in enumerate(df.iterrows()):
        with Image.open(row["path"]) as im:
            im_rgb = im.convert("RGB")
            x = tfm(im_rgb).unsqueeze(0).to(device)
            disp_img = np.asarray(im_rgb.resize((disp, disp), Image.LANCZOS)) / 255.0
        model.zero_grad()
        logits = model(x)
        cls = int(logits.argmax(1))
        logits[0, cls].backward()

        A, G = acts["a"][0], grads["g"][0]              # (C,h,w)
        w = G.mean(dim=(1, 2))                            # GAP weights
        cam = torch.relu((w[:, None, None] * A).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(cam[None, None], size=(disp, disp),
                                              mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        axes[0][j].imshow(disp_img, interpolation="lanczos")
        axes[0][j].set_title(f"y={int(row['label'])}  pred={cls}", fontsize=11)
        axes[1][j].imshow(disp_img, interpolation="lanczos")
        axes[1][j].imshow(cam, cmap="Blues_r", alpha=0.55, interpolation="bilinear")
        for r in (0, 1):
            axes[r][j].set_xticks([]); axes[r][j].set_yticks([])
            for sp in axes[r][j].spines.values():
                sp.set_visible(False)
    axes[0][0].set_ylabel("Input cell", fontsize=12, fontweight="bold")
    axes[1][0].set_ylabel("Saliency overlay", fontsize=12, fontweight="bold")
    fig.suptitle("Backbone class-discriminative saliency (PCMMD test cells)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); out = figures / "fig_gradcam.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(figures / "fig_gradcam.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam] wrote {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Backbone Grad-CAM (Phase 9).")
    add_common_args(p)
    p.add_argument("--model", default="cnn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=6)
    args = p.parse_args()
    cfg = load_config(args.config or ["configs/backbones.yaml"])
    gradcam(cfg, model_name=args.model, seed=args.seed, n=args.n)


if __name__ == "__main__":
    main()
