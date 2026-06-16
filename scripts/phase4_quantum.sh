#!/usr/bin/env bash
# Phase 4 — precompute frozen-backbone features, then train the quantum head.
# Needs Phase 2 splits. Usage: scripts/phase4_quantum.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE="${1:-}"

echo "[phase4] precomputing features (frozen backbone -> PCA -> [-pi,pi])..."
python -m src.features.precompute --config configs/backbones.yaml configs/quantum.yaml ${SMOKE}

echo "[phase4] training quantum head (ablation + main vs param-matched MLP)..."
python -m src.train.train_quantum --config configs/backbones.yaml configs/quantum.yaml ${SMOKE}

echo "[phase4] done. See results/quantum_ablation.csv, results/quantum_main.csv"
