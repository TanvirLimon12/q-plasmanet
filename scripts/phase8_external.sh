#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
python -m src.eval.external_validation --config configs/backbones.yaml configs/quantum.yaml --datasets bloodmnist pathmnist --classes 0 1 "${1:-}"
