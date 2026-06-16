#!/usr/bin/env bash
set -euo pipefail; cd "$(dirname "$0")/.."
python -m src.viz.plots --config configs/base.yaml
python -m src.viz.gradcam --config configs/backbones.yaml --seed 0 --n 6
