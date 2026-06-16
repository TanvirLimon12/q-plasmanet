#!/usr/bin/env bash
# Phase 1 — download (best-effort) + audit PCMMD.
# Usage: scripts/phase1_audit.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE="${1:-}"

echo "[phase1] attempting download (fails loudly if manual step needed)..."
python -m src.data.download --config configs/base.yaml || \
  echo "[phase1] download not completed automatically — see message above; continuing to audit."

echo "[phase1] running audit..."
python -m src.data.audit --config configs/base.yaml ${SMOKE}

echo "[phase1] done. See results/dataset_summary.csv and results/audit_log.txt"
