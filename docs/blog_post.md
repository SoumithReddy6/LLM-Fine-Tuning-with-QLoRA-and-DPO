# From Zero-Shot Prompts to Preference-Tuned Financial LLMs

## Summary

This project fine-tunes open LLMs for financial sentiment classification using QLoRA and then improves output preference alignment with DPO. The goal is to turn a generic instruction model into a domain-specialized analyst assistant that returns calibrated, label-valid sentiment judgments for financial text.

## Why Financial Sentiment

Financial language is compact, high-context, and easy to misread. Words like "liability", "charge", "guidance", and "beat" have domain-specific meaning, and a general model can produce plausible explanations while missing the actual sentiment label. That makes it a useful fine-tuning target for applied AI roles.

## Method

The workflow builds instruction records from FLUE-style sentiment examples, trains 4-bit QLoRA adapters for Llama 3 8B and Qwen2.5 7B, and then converts labeled examples into chosen/rejected preference pairs for DPO. Experiments are tracked in MLflow with optional Weights & Biases logging.

## Evaluation

The harness reports accuracy, macro-F1, calibration error, Brier score, invalid-label rate, hallucination proxy rate, latency, and lift over a zero-shot GPT-4-style baseline. The project also includes a deterministic smoke test so the evaluation path can run locally before GPU training.

## Expected Findings

QLoRA should improve task accuracy by teaching the model the label space and financial phrasing. DPO should reduce invalid or over-explained outputs by rewarding concise answers that match the required schema. Ablations across rank, learning rate, and data mixture should reveal whether performance is adapter-capacity-limited or data-quality-limited.

## Reproducibility

The repository publishes model configs, sweep configs, training scripts, DPO scripts, evaluation scripts, and a blog-ready writeup so another engineer can rerun the project on Kaggle or Colab.
