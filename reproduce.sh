#!/usr/bin/env bash
# One-command reproduction of the full Q-PlasmaNet pipeline (real PCMMD required in
# data/raw/pcmmd; see README Phase 1). Classical runs use the GPU (CUDA/MPS) if
# available; quantum stays on CPU. Quantum-heavy phases are the time sink.
set -euo pipefail
cd "$(dirname "$0")"
CFG="configs/backbones.yaml configs/quantum.yaml"
NOISE="configs/noise.yaml"
SEEDS="0 1 2 3 4"

python -m src.data.audit  --config configs/base.yaml
python -m src.data.splits --config configs/base.yaml

# Phase 3 — strong baselines (GPU)
python -m src.train.train_classical --config configs/backbones.yaml --seeds $SEEDS

# Phase 4 — trained-cnn backbone -> features -> quantum head (5-seed) + ablation
python -m src.features.precompute  --config $CFG --backbones cnn --dims 24 --seeds $SEEDS --from-classifier cnn
python -m src.train.train_quantum  --config $CFG --main-only --seeds $SEEDS

# Phases 5-7
python -m src.eval.robustness   --config $CFG --models cnn q_plasmanet --seeds $SEEDS
python -m src.eval.nisq         --config $CFG $NOISE --seeds $SEEDS --limit 120
python -m src.eval.patient_agg  --config configs/backbones.yaml --seed 0 --model cnn

# Phase 8 — external validation
python -m src.eval.external_validation --config $CFG --datasets bloodmnist pathmnist --classes 0 1

# Q1 analyses
python -m src.eval.trainability --config configs/quantum.yaml
python -m src.eval.expressivity --config configs/quantum.yaml
python -m src.eval.effective_dim --config $CFG
python -m src.eval.separability --config $CFG
python -m src.eval.data_scaling --config $CFG

# Phase 9 — figures, tables, stats
python -m src.viz.plots            --config configs/base.yaml
python -m src.viz.gradcam          --config configs/backbones.yaml --seed 0
python -m src.viz.calibration_plot --config $CFG --seed 0
python -m src.viz.tables           --config configs/base.yaml
python -m src.utils.make_stats     --config configs/base.yaml

echo "[reproduce] done — see results/, figures/, docs/"
