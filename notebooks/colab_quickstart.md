# Colab Quickstart — Run a REAL fine-tune for free

This is the exact, ordered path to fine-tune a small model end-to-end on a free
Colab T4 GPU and produce **real** metrics. Each cell has a plain-English note on
what it does and why. Copy each block into its own Colab cell, in order.

**Before you start:** In Colab, go to `Runtime → Change runtime type → T4 GPU`.

---

### Cell 1 — Confirm you have a GPU
*Why: training needs a GPU. If this prints "no GPU", fix the runtime type first.*

```python
!nvidia-smi
```

### Cell 2 — Get the code
*Why: pulls your repository into Colab so the scripts and configs are available.*

```python
!git clone https://github.com/SoumithReddy6/LLM-Fine-Tuning-with-QLoRA-and-DPO.git
%cd LLM-Fine-Tuning-with-QLoRA-and-DPO
```

### Cell 3 — Install a known-good, compatible stack
*Why: floating versions break. These pins are mutually compatible and match the
training scripts' API, so the run works on the first try.*

```python
!pip -q install \
  "torch==2.4.1" \
  "transformers==4.46.3" \
  "trl==0.11.4" \
  "peft==0.13.2" \
  "accelerate==1.1.1" \
  "datasets==3.1.0" \
  "bitsandbytes==0.44.1"
```

### Cell 4 — Build the real dataset
*Why: turns the Financial PhraseBank benchmark into train / test / preference
files. The test set is held out, so your accuracy is measured on data the model
never trained on — that's what makes the number real.*

```python
!python3 scripts/prepare_financial_phrasebank.py --max-train 2000 --test-size 0.2 --seed 7
```

### Cell 5 — Measure the BASELINE first (base model, no fine-tuning)
*Why: to prove fine-tuning helped, you need a "before" number. This runs the raw
base model zero-shot on the test set. Expect modest accuracy and some messy /
invalid outputs — that's the point.*

```python
!python3 scripts/predict.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --input data/financial_test.jsonl \
  --output artifacts/baseline_preds.jsonl \
  --load-in-4bit
```

### Cell 6 — Supervised fine-tune (QLoRA / SFT)
*Why: this is the actual training. It teaches the model to output clean
financial-sentiment labels. On a T4 with 2,000 examples this takes a few
minutes. You'll see the loss printed as it learns.*

```python
!python3 scripts/train_sft.py \
  --config configs/experiments/qwen25_0_5b_qlora_rank16.yaml \
  --local-data data/financial_train.jsonl
```

### Cell 7 — Get the fine-tuned model's predictions
*Why: same script as the baseline, but now with `--adapter` pointing at what you
just trained. This is the "after" number.*

```python
!python3 scripts/predict.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/qwen25_0_5b_qlora_rank16 \
  --input data/financial_test.jsonl \
  --output artifacts/tuned_preds.jsonl \
  --load-in-4bit
```

### Cell 8 — Score it: real accuracy, F1, calibration, and the lift over baseline
*Why: computes the honest metrics and the improvement fine-tuning bought. The
`relative_accuracy_lift` field is your headline result.*

```python
!python3 scripts/evaluate_model.py --config configs/experiments/eval_qwen25_0_5b.yaml
```

---

## Optional: DPO (preference tuning)

DPO further nudges the model toward preferred answers. Run these after the steps
above.

### Cell 9 — Train DPO from the SFT adapter
```python
!python3 scripts/train_dpo.py --config configs/experiments/qwen25_0_5b_dpo.yaml
```

### Cell 10 — Predict + score the DPO model
```python
!python3 scripts/predict.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/qwen25_0_5b_dpo \
  --input data/financial_test.jsonl \
  --output artifacts/dpo_preds.jsonl \
  --load-in-4bit

import json
from pathlib import Path
import sys
sys.path.insert(0, "src")
from qlora_dpo_finance.data import read_jsonl
from qlora_dpo_finance.metrics import evaluate_predictions
metrics = evaluate_predictions(read_jsonl("artifacts/dpo_preds.jsonl"), read_jsonl("artifacts/baseline_preds.jsonl"))
print(json.dumps(metrics, indent=2))
```

### Cell 11 — Save your real results back to the repo
*Why: download the metrics so you can commit them. These replace the fake
hand-authored smoke numbers with results a model actually produced.*

```python
from google.colab import files
files.download("artifacts/eval_qwen25_0_5b.json")
files.download("artifacts/tuned_preds.jsonl")
```

---

## Scaling up later

The 7B/8B configs (`configs/experiments/qwen25_7b_*.yaml`, `llama3_8b_*.yaml`)
use the **identical method** — only the model size and a larger GPU differ. Once
the small run works, swapping the model name is the only change. For Llama 3,
request model access on Hugging Face and run `huggingface-cli login` first.

## If a cell errors

- **"CUDA out of memory"** — lower `per_device_train_batch_size` in the config
  (try 4, then 2), or `--max-train 1000` in Cell 4.
- **A `TypeError` from `SFTTrainer`/`DPOTrainer`** — a TRL version mismatch.
  Re-run Cell 3 exactly (the pins matter), then `Runtime → Restart session`.
- **Anything else** — copy the full error text and send it over; debugging a real
  training run is normal, and it's exactly the experience interviewers ask about.
