.PHONY: install test smoke manifest

install:
	pip install -r requirements.txt

test:
	pytest

smoke:
	python3 scripts/run_smoke_eval.py --predictions data/samples/predictions_sample.jsonl

manifest:
	python3 scripts/generate_ablation_manifest.py --output artifacts/ablation_manifest.csv
