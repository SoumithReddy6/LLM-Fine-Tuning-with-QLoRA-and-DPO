# Results

Real fine-tuning run on a free Colab T4 GPU.

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Method:** QLoRA (4-bit NF4, double-quant, bf16 compute) + LoRA rank-16, 1 epoch
- **Dataset:** Financial PhraseBank (`sentences_50agree`), ~2,000 train (capped via `--max-train`) / 970 held-out test
- **Reproduce:** `notebooks/colab_quickstart.ipynb`

## Headline

| Metric | Fine-tuned (QLoRA SFT) | Base, zero-shot |
| --- | ---: | ---: |
| Accuracy | **80.8%** | 25.4% |
| Macro-F1 | 0.803 | — |
| Expected calibration error (ECE) | 0.044 | — |
| Brier score | 0.310 | — |
| Invalid-output rate | 0.3% | — |
| Hallucination-proxy rate | 0.3% | — |
| Mean latency / example | 336 ms | — |

Relative accuracy lift over the zero-shot base model: ~3.2x. The headline framing
to use is the **absolute** improvement (25% → 81%), not the relative multiple —
the base score is low mainly because a 0.5B base model does not follow the
required label format without tuning, so the lift reflects both task learning and
format discipline.

## Full metrics (eval on 970 held-out examples)

```json
{
  "examples": 970,
  "accuracy": 0.8082,
  "macro_f1": 0.8027,
  "expected_calibration_error": 0.044,
  "brier_score": 0.3097,
  "invalid_label_rate": 0.0031,
  "hallucination_proxy_rate": 0.0031,
  "mean_latency_ms": 336.13,
  "confusion_matrix": {
    "negative": { "negative": 102, "neutral": 16,  "positive": 7   },
    "neutral":  { "negative": 11,  "neutral": 479, "positive": 72  },
    "positive": { "negative": 4,   "neutral": 73,  "positive": 203 }
  },
  "label_distribution": { "negative": 126, "neutral": 564, "positive": 280 },
  "baseline_accuracy": 0.2536,
  "relative_accuracy_lift": 2.1869
}
```

## Error analysis

The dominant error mode is **neutral ↔ positive** confusion (72 neutral predicted
positive, 73 positive predicted neutral). Negative is the cleanest class. This is
expected on Financial PhraseBank, where the neutral/positive boundary is the most
subjective. Next improvements to try: class-balanced sampling (neutral is the
majority at 564/970), and DPO preference tuning on the confused pairs.

## Not yet run

- DPO preference tuning (script and config are included; the SFT result above is
  SFT-only).
- 7B / 8B scale-ups (configs included; require larger GPUs).
