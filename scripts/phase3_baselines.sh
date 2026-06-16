#!/usr/bin/env bash
# Phase 3 — classical baselines (needs Phase 2 splits to exist).
# Usage: scripts/phase3_baselines.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.train.train_classical --config configs/backbones.yaml "${1:-}"

echo "[phase3] done. See results/baselines.csv (mean±std) and results/baselines_raw.csv"
