# Q-PlasmaNet

Hybrid classical-quantum image classifier for plasma cell recognition (multiple
myeloma), built on the PCMMD dataset. A frozen CNN backbone extracts features on
GPU; a small variational quantum circuit (4-8 qubits, PennyLane) sits on top as the
classifier head and trains on CPU. The point was to test, honestly, whether a NISQ
quantum head can match strong classical baselines at the same parameter budget --
not to claim quantum advantage.

Short answer from the runs in `results/`: it reaches parity. The re-uploading VQC
head gets F1 ~0.927 with 178 trainable parameters, statistically tied with a
parameter-matched MLP (~0.932); the best classical CNN (MobileNetV2) is higher at
0.949 but uses 2.2M parameters.

## Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
python -c "import torch; print('cuda', torch.cuda.is_available(), 'mps', torch.backends.mps.is_available())"
```

PennyLane (`lightning.qubit` for analytic training, `default.mixed` for the noise
study). Backbone training uses CUDA or Apple MPS; the quantum head stays on CPU.

## Reproducing

The whole pipeline is split into phases. Run them in order, or use the runner:

```bash
bash reproduce.sh          # full pipeline, seeds 0-4
# or individual phases:
bash scripts/phase3_baselines.sh
bash scripts/phase4_quantum.sh
bash scripts/phase7_patient.sh
```

A few standing rules baked into the code:

- The backbone is never trained with the quantum head unless you pass `--joint`.
- Everything runs over seeds `[0,1,2,3,4]` and reports mean +/- std.
- Splits are patient-level -- no patient appears in two splits.
- Every script takes `--smoke` for a 1-seed, tiny-subset run (<2 min) before you
  commit to a full run.

Note: Apple MPS is not bit-deterministic, so classical metrics can wobble at the
~0.001 level between an MPS and a CPU run. That's inside the reported std.

## Layout

```
src/models/      backbones, classical heads, quantum head (VQC + data re-uploading)
src/features/    precompute frozen backbone features
src/train/       training engine, classical + quantum trainers
src/eval/        calibration, robustness, NISQ noise, patient aggregation, few-shot
src/viz/         plots, tables, Grad-CAM
configs/         base / backbones / quantum / noise YAMLs
scripts/         phase1..phase9 runners
tests/           per-phase tests
results/         metric CSVs (summaries)
figures/         generated figures
```

PCMMD is CC BY 4.0; download it separately and point the configs at it. Data,
checkpoints, and any `apikey.json` are gitignored.
