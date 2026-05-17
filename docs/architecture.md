# Architecture

## Flow

1. Load financial sentiment records from Hugging Face or local JSONL fixtures.
2. Convert records into instruction-tuning examples.
3. Train a 4-bit QLoRA adapter with rank-16 defaults.
4. Generate chosen/rejected preference pairs.
5. Run DPO using the SFT adapter as the starting policy.
6. Evaluate the final adapter against baseline predictions.
7. Track metrics and parameters in MLflow and optionally Weights & Biases.

## Core Modules

- `config.py`: YAML config loading and validation.
- `data.py`: dataset normalization, prompt formatting, and preference pair creation.
- `metrics.py`: accuracy, macro-F1, calibration, hallucination proxy, and lift.
- `tracking.py`: MLflow and W&B logging helpers.
- `trainer.py`: QLoRA, tokenizer, and PEFT setup utilities.

## GPU Notes

The training scripts expect CUDA for 7B/8B models. Local CPU execution is supported only for smoke tests and metric validation.
