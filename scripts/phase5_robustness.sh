#!/usr/bin/env bash
# Phase 5 — robustness suite (inference only). Needs Phase 3 + Phase 4 checkpoints.
# Usage: scripts/phase5_robustness.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.eval.robustness --config configs/backbones.yaml configs/quantum.yaml \
  --models cnn mobilenet_v2 resnet18 efficientnet_b0 cnn_mlp q_plasmanet "${1:-}"

echo "[phase5] done. See results/robustness.csv and results/robustness_score.csv"
