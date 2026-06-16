#!/usr/bin/env bash
# Phase 6 — NISQ shot/noise evaluation (the time sink). Needs Phase 4 main checkpoint.
# Usage: scripts/phase6_nisq.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.eval.nisq --config configs/backbones.yaml configs/quantum.yaml configs/noise.yaml "${1:-}"

echo "[phase6] done. See results/nisq.csv"
