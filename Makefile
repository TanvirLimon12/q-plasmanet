.PHONY: help test smoke audit splits baselines quantum analyses figures repro clean

help:
	@echo "test      - run all smoke tests"
	@echo "audit     - PCMMD data audit (Phase 1)"
	@echo "splits    - build patient-level/stratified splits (Phase 2)"
	@echo "baselines - classical baselines, 5 seeds (Phase 3, GPU)"
	@echo "quantum   - precompute + quantum main, 5 seeds (Phase 4)"
	@echo "analyses  - trainability/eff-dim/expressivity/separability/data-scaling (Q1)"
	@echo "figures   - plots + gradcam + calibration + tables + stats"
	@echo "repro     - full pipeline (reproduce.sh)"

CFG = configs/backbones.yaml configs/quantum.yaml

test:
	@for t in tests/test_*.py; do echo "== $$t =="; python3 $$t || exit 1; done

audit:    ; python -m src.data.audit  --config configs/base.yaml
splits:   ; python -m src.data.splits --config configs/base.yaml
baselines:; python -m src.train.train_classical --config configs/backbones.yaml --seeds 0 1 2 3 4
quantum:
	python -m src.features.precompute --config $(CFG) --backbones cnn --dims 24 --seeds 0 1 2 3 4 --from-classifier cnn
	python -m src.train.train_quantum --config $(CFG) --main-only --seeds 0 1 2 3 4

analyses:
	python -m src.eval.trainability  --config configs/quantum.yaml
	python -m src.eval.expressivity  --config configs/quantum.yaml
	python -m src.eval.effective_dim --config $(CFG)
	python -m src.eval.separability  --config $(CFG)
	python -m src.eval.data_scaling  --config $(CFG)

figures:
	python -m src.viz.plots            --config configs/base.yaml
	python -m src.viz.gradcam          --config configs/backbones.yaml --seed 0
	python -m src.viz.calibration_plot --config $(CFG) --seed 0
	python -m src.viz.tables           --config configs/base.yaml
	python -m src.utils.make_stats     --config configs/base.yaml

repro: ; bash reproduce.sh

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
