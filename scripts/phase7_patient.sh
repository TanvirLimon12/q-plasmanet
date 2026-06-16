#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
python -m src.eval.patient_agg --config configs/backbones.yaml --seed "${1:-0}" --model "${2:-cnn}"
