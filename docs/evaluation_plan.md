# Evaluation Plan

The benchmark harness measures eight task and reliability metrics for financial sentiment classification.

## Metrics

1. Accuracy
2. Macro-F1
3. Expected calibration error
4. Brier score
5. Invalid-label rate
6. Hallucination proxy rate
7. Mean latency
8. Relative lift over baseline

## Baselines

- Zero-shot GPT-4-style baseline from `scripts/run_openai_baseline.py`.
- Deterministic lexical baseline for local smoke tests.
- Pre-DPO SFT adapter versus post-DPO adapter.

## Acceptance Targets

- 20%+ relative improvement over zero-shot GPT-4 baseline on the selected financial sentiment test set.
- 100% valid label format in production evaluation.
- Lower calibration error after DPO than after SFT alone.
- 50+ MLflow runs logged across model, learning-rate, LoRA-rank, seed, and data-mixture ablations.

## Reporting

Final reports should include:

- Dataset version and split.
- Model checkpoint and adapter path.
- Quantization and LoRA settings.
- All eight metrics.
- Confusion matrix.
- Failure examples for hallucination or invalid labels.
